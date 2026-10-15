//! Prefill EXL3 linears against the fork, on real 27B layer-0 and layer-3 weights.
//!
//! Rows 2, 47, 144, 145, and 1499. Gate, up, and down on both layers, layer-0
//! `out_proj`, layer-3 `o_proj`, GDN qkv+z mgemm, and attention qkv-gate mgemm.
//! The fork reference is `tests/prefill_lin_ref.py` (`LinearEXL3.forward` and
//! `ext.exl3_mgemm`). Inputs are regenerated from a seed. The coop autotune
//! file is snapshotted and restored.
//!
//! Skips unless `BERTH_GPU_TESTS=1`. Also skips when the checkpoint is missing
//! or port 8889 is listening.

use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use berth_cuda::{Context, DeviceBuffer, Stream};
use berth_exl3::{Bitrate, Checkpoint, HadamardScale, LinearExl3, TensorInfo};
use berth_kernels::ops;
use berth_kernels::prefill_lin;
use berth_kernels::{Codebook, OutDtype};

const ROWS: [i32; 5] = [2, 47, 144, 145, 1499];

fn model_dir() -> String {
    std::env::var("BERTH_27B_DIR")
        .unwrap_or_else(|_| "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3".to_string())
}

fn serve_listening() -> bool {
    let output = match Command::new("ss").args(["-ltn"]).output() {
        Ok(output) => output,
        Err(_) => return true,
    };
    if !output.status.success() {
        return true;
    }
    String::from_utf8_lossy(&output.stdout).contains(":8889 ")
}

fn tune_cache() -> PathBuf {
    if let Ok(override_path) = std::env::var("EXLLAMAV3_TUNE_CACHE") {
        if !override_path.is_empty() {
            let path = PathBuf::from(override_path);
            if path.is_dir() {
                return path.join("coop_autotune_v1.bin");
            }
            return path;
        }
    }
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("exllamav3/autotune/coop_autotune_v1.bin");
        }
    }
    let home = std::env::var("HOME").expect("HOME");
    PathBuf::from(home).join(".cache/exllamav3/autotune/coop_autotune_v1.bin")
}

struct CacheGuard {
    path: PathBuf,
    original: Vec<u8>,
}

impl CacheGuard {
    fn snapshot() -> Self {
        let path = tune_cache();
        let original =
            std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
        Self { path, original }
    }
}

impl Drop for CacheGuard {
    fn drop(&mut self) {
        if std::fs::read(&self.path).ok().as_deref() != Some(self.original.as_slice()) {
            if let Err(err) = std::fs::write(&self.path, &self.original) {
                eprintln!("failed to restore {}: {err}", self.path.display());
            } else {
                eprintln!("restored {}", self.path.display());
            }
        }
    }
}

fn read_tensor(info: &TensorInfo) -> Vec<u8> {
    let mut file =
        File::open(&info.shard).unwrap_or_else(|err| panic!("{}: {err}", info.shard.display()));
    file.seek(SeekFrom::Start(info.offset))
        .unwrap_or_else(|err| panic!("seek {}: {err}", info.name));
    let mut bytes = vec![0u8; info.length as usize];
    file.read_exact(&mut bytes)
        .unwrap_or_else(|err| panic!("read {}: {err}", info.name));
    bytes
}

fn unpacked(scale: &HadamardScale) -> &TensorInfo {
    match scale {
        HadamardScale::Unpacked(info) => info,
        HadamardScale::Packed(_) => panic!("27B linears store fp16 suh / svh"),
    }
}

fn linear<'a>(ckpt: &'a Checkpoint, prefix: &str) -> &'a LinearExl3 {
    ckpt.linears()
        .iter()
        .find(|item| item.prefix == prefix)
        .unwrap_or_else(|| panic!("{prefix} is not an EXL3 linear"))
}

fn require_mul1(item: &LinearExl3) {
    let Bitrate::Int(k_bits) = item.k else {
        panic!("{}: half-integer rate", item.prefix);
    };
    assert_eq!(k_bits, 4, "{}", item.prefix);
    assert!(
        matches!(item.codebook, berth_exl3::Codebook::Mul1),
        "{}",
        item.prefix
    );
    assert!(item.bias.is_none(), "{} has a bias", item.prefix);
}

fn shape_kn(item: &LinearExl3) -> (i32, i32) {
    let shape = &item.trellis.shape;
    assert_eq!(shape.len(), 3, "{}", item.prefix);
    let k_in = i32::try_from(shape[0] * 16).unwrap();
    let n = i32::try_from(shape[1] * 16).unwrap();
    assert_eq!(shape[2] / 16, 4, "{}", item.prefix);
    (k_in, n)
}

fn fill_f16(seed: u64, n: usize) -> Vec<u8> {
    let mut state = seed;
    let mut out = vec![0u8; n * 2];
    for slot in out.chunks_exact_mut(2) {
        state = state
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x6A09_E667_F3BC_C909);
        let mag = ((state >> 48) as u16) & 0x03FF;
        let sign = ((state >> 47) as u16) & 0x8000;
        slot.copy_from_slice(&(sign | 0x3C00 | mag).to_le_bytes());
    }
    out
}

fn differ(got: &[u8], want: &[u8], elem: usize) -> usize {
    assert_eq!(got.len(), want.len());
    got.chunks(elem)
        .zip(want.chunks(elem))
        .filter(|(g, w)| g != w)
        .count()
}

struct Spec {
    id: String,
    kind: &'static str,
    prefixes: Vec<String>,
    rows: i32,
    out: OutDtype,
}

fn specs() -> Vec<Spec> {
    let mut out = Vec::new();
    let l0 = "model.language_model.layers.0";
    let l3 = "model.language_model.layers.3";
    let linears = [
        ("l0-gate", format!("{l0}.mlp.gate_proj"), OutDtype::F16),
        ("l0-up", format!("{l0}.mlp.up_proj"), OutDtype::F16),
        ("l0-down", format!("{l0}.mlp.down_proj"), OutDtype::F32),
        (
            "l0-out",
            format!("{l0}.linear_attn.out_proj"),
            OutDtype::F32,
        ),
        ("l3-gate", format!("{l3}.mlp.gate_proj"), OutDtype::F16),
        ("l3-up", format!("{l3}.mlp.up_proj"), OutDtype::F16),
        ("l3-down", format!("{l3}.mlp.down_proj"), OutDtype::F32),
        ("l3-o", format!("{l3}.self_attn.o_proj"), OutDtype::F32),
    ];
    for rows in ROWS {
        for (name, prefix, dtype) in &linears {
            out.push(Spec {
                id: format!("{name}-r{rows}"),
                kind: "linear",
                prefixes: vec![prefix.clone()],
                rows,
                out: *dtype,
            });
        }
        out.push(Spec {
            id: format!("l0-qkvz-r{rows}"),
            kind: "mgemm",
            prefixes: vec![
                format!("{l0}.linear_attn.in_proj_qkv"),
                format!("{l0}.linear_attn.in_proj_z"),
            ],
            rows,
            out: OutDtype::F32,
        });
        out.push(Spec {
            id: format!("l3-qkv-r{rows}"),
            kind: "mgemm",
            prefixes: vec![
                format!("{l3}.self_attn.q_proj"),
                format!("{l3}.self_attn.k_proj"),
                format!("{l3}.self_attn.v_proj"),
            ],
            rows,
            out: OutDtype::F16,
        });
    }
    out
}

fn nvidia_ld_path() -> String {
    let base = PathBuf::from("/home/sam/openwebui/.venv/lib/python3.13/site-packages/nvidia");
    let mut libs = Vec::new();
    let entries =
        std::fs::read_dir(&base).unwrap_or_else(|err| panic!("read {}: {err}", base.display()));
    for entry in entries {
        let lib = entry.expect("nvidia entry").path().join("lib");
        if lib.is_dir() {
            libs.push(lib);
        }
    }
    libs.sort();
    let mut path = libs
        .iter()
        .map(|lib| lib.display().to_string())
        .collect::<Vec<_>>()
        .join(":");
    if let Ok(existing) = std::env::var("LD_LIBRARY_PATH") {
        if !existing.is_empty() {
            path.push(':');
            path.push_str(&existing);
        }
    }
    path
}

fn run_python(manifest: &Path) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/prefill_lin_ref.py");
    let python = "/home/sam/qwen38-exl3-dflash2/env/bin/python";
    let mut cmd = Command::new(python);
    cmd.arg(&script)
        .arg(manifest)
        .env("PYTHONPATH", "/home/sam/qwen38-exl3-dflash2/exllamav3")
        .env("TRITON_LIBCUDA_PATH", "/usr/lib/x86_64-linux-gnu")
        .env("LD_LIBRARY_PATH", nvidia_ld_path())
        .env("EXL3_HGEMM_F16ACC", "auto")
        .env("EXL3_INT8_GEMV", "2")
        .env("EXL3_INT8_GEMV_MAX_K", "5")
        .env("EXL3_QC_STAGING", "1");
    let status = cmd
        .status()
        .unwrap_or_else(|err| panic!("spawn {python}: {err}"));
    if !status.success() {
        panic!("prefill reference failed ({status})");
    }
}

fn dev_ptr(buf: &DeviceBuffer<u8>) -> *mut c_void {
    buf.ptr().as_u64() as *mut c_void
}

fn upload(stream: &Stream, bytes: &[u8]) -> DeviceBuffer<u8> {
    DeviceBuffer::from_host(stream, bytes).unwrap()
}

struct Weights {
    k_in: i32,
    n: i32,
    trellis: DeviceBuffer<u8>,
    suh: DeviceBuffer<u8>,
    svh: DeviceBuffer<u8>,
}

fn load_weights(stream: &Stream, item: &LinearExl3) -> Weights {
    require_mul1(item);
    let (k_in, n) = shape_kn(item);
    let suh = unpacked(&item.input_scale);
    let svh = unpacked(&item.output_scale);
    assert_eq!(suh.length, k_in as u64 * 2, "{}", item.prefix);
    assert_eq!(svh.length, n as u64 * 2, "{}", item.prefix);
    Weights {
        k_in,
        n,
        trellis: upload(stream, &read_tensor(&item.trellis)),
        suh: upload(stream, &read_tensor(suh)),
        svh: upload(stream, &read_tensor(svh)),
    }
}

fn run_linear_case(stream: &Stream, ckpt: &Checkpoint, spec: &Spec, input: &[u8], out_dir: &Path) {
    let item = linear(ckpt, &spec.prefixes[0]);
    let weights = load_weights(stream, item);
    let elem = match spec.out {
        OutDtype::F16 => 2,
        OutDtype::F32 => 4,
    };
    let rows = spec.rows as usize;
    assert_eq!(input.len(), rows * weights.k_in as usize * 2);
    let a = upload(stream, input);
    let a_had = DeviceBuffer::<u8>::zeroed(stream, input.len()).unwrap();
    let c = DeviceBuffer::<u8>::zeroed(stream, rows * weights.n as usize * elem).unwrap();
    let scratch_cols = weights.n.min(32768) as usize;
    let scratch = if spec.rows > 144 {
        Some(DeviceBuffer::<u8>::zeroed(stream, weights.k_in as usize * scratch_cols * 2).unwrap())
    } else {
        None
    };
    let scratch_ptr = scratch
        .as_ref()
        .map(dev_ptr)
        .unwrap_or(std::ptr::null_mut());
    unsafe {
        prefill_lin::exl3_linear(
            dev_ptr(&a),
            dev_ptr(&weights.trellis),
            dev_ptr(&c),
            dev_ptr(&weights.suh),
            dev_ptr(&a_had),
            dev_ptr(&weights.svh),
            scratch_ptr,
            spec.rows,
            weights.k_in,
            weights.n,
            4,
            Codebook::Mul1,
            spec.out,
            stream.handle().as_ptr(),
        )
    }
    .unwrap_or_else(|err| panic!("{}: {err}", spec.id));
    let mut got = vec![0u8; c.len()];
    c.copy_to_host(stream, &mut got).unwrap();
    let want = std::fs::read(out_dir.join(format!("{}.out", spec.id)))
        .unwrap_or_else(|err| panic!("read {}: {err}", spec.id));
    let diffs = differ(&got, &want, elem);
    assert_eq!(
        diffs,
        0,
        "{}: {diffs} elements differ ({})",
        spec.id,
        prefill_lin::last_launch()
    );
    eprintln!("{} {}", spec.id, prefill_lin::last_launch());
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}

fn run_mgemm_case(stream: &Stream, ckpt: &Checkpoint, spec: &Spec, input: &[u8], out_dir: &Path) {
    let elem = match spec.out {
        OutDtype::F16 => 2usize,
        OutDtype::F32 => 4usize,
    };
    let loaded: Vec<Weights> = spec
        .prefixes
        .iter()
        .map(|prefix| load_weights(stream, linear(ckpt, prefix)))
        .collect();
    let k_in = loaded[0].k_in;
    assert!(loaded.iter().all(|item| item.k_in == k_in));
    let width = loaded.iter().fold(
        0,
        |acc, item| {
            if acc == 0 {
                item.n
            } else {
                gcd(acc, item.n)
            }
        },
    );
    assert!(width % 128 == 0 && width >= 256, "{}", spec.id);

    let rows = spec.rows as usize;
    assert_eq!(input.len(), rows * k_in as usize * 2);
    let mut outputs = Vec::new();
    for item in &loaded {
        outputs.push(DeviceBuffer::<u8>::zeroed(stream, rows * item.n as usize * elem).unwrap());
    }

    let mut b_host = Vec::new();
    let mut svh_host = Vec::new();
    let mut suh_host = Vec::new();
    let mut c_host = Vec::new();
    let mut size_n = Vec::new();
    let mut strides = Vec::new();
    let mut had_src = Vec::new();
    for (index, item) in loaded.iter().enumerate() {
        suh_host.push(item.suh.ptr().as_u64());
        let mut n0 = 0;
        while n0 < item.n {
            let trellis_off = (n0 / 16) * 16 * 4 * 2;
            b_host.push(item.trellis.ptr().as_u64() + trellis_off as u64);
            svh_host.push(item.svh.ptr().as_u64() + n0 as u64 * 2);
            c_host.push(outputs[index].ptr().as_u64() + n0 as u64 * elem as u64);
            size_n.push(width);
            strides.push(item.n);
            had_src.push(index as i32);
            n0 += width;
        }
    }
    let slices = had_src.len() as i32;
    let b_list = DeviceBuffer::from_host(stream, &b_host).unwrap();
    let suh_list = DeviceBuffer::from_host(stream, &suh_host).unwrap();
    let svh_list = DeviceBuffer::from_host(stream, &svh_host).unwrap();
    let c_ptrs = DeviceBuffer::from_host(stream, &c_host).unwrap();
    let size_n_list = DeviceBuffer::from_host(stream, &size_n).unwrap();
    let n_stride_list = DeviceBuffer::from_host(stream, &strides).unwrap();
    let had_src_list = DeviceBuffer::from_host(stream, &had_src).unwrap();
    let a = upload(stream, input);
    let a_had = DeviceBuffer::<u8>::zeroed(stream, suh_host.len() * input.len()).unwrap();
    let carrier =
        DeviceBuffer::<u8>::zeroed(stream, slices as usize * rows * width as usize * elem).unwrap();

    unsafe {
        prefill_lin::exl3_mgemm(
            dev_ptr(&a),
            b_list.ptr().as_u64() as *const c_void,
            dev_ptr(&carrier),
            suh_list.ptr().as_u64() as *const c_void,
            dev_ptr(&a_had),
            svh_list.ptr().as_u64() as *const c_void,
            spec.rows,
            k_in,
            width,
            4,
            Codebook::Mul1,
            spec.out,
            1,
            slices,
            size_n_list.ptr().as_u64() as *const i32,
            c_ptrs.ptr().as_u64() as *const c_void,
            n_stride_list.ptr().as_u64() as *const i32,
            had_src_list.ptr().as_u64() as *const i32,
            suh_host.len() as i32,
            stream.handle().as_ptr(),
        )
    }
    .unwrap_or_else(|err| panic!("{}: {err}", spec.id));

    for (index, output) in outputs.iter().enumerate() {
        let mut got = vec![0u8; output.len()];
        output.copy_to_host(stream, &mut got).unwrap();
        let want = std::fs::read(out_dir.join(format!("{}.{index}.out", spec.id)))
            .unwrap_or_else(|err| panic!("read {}.{index}: {err}", spec.id));
        let diffs = differ(&got, &want, elem);
        assert_eq!(
            diffs,
            0,
            "{} source {index}: {diffs} elements differ ({})",
            spec.id,
            ops::last_launch()
        );
    }
    eprintln!("{} {}", spec.id, ops::last_launch());
}

#[test]
fn prefill_matches_fork() {
    if std::env::var("BERTH_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping prefill_lin: set BERTH_GPU_TESTS=1");
        return;
    }
    if serve_listening() {
        eprintln!("skipping prefill_lin: port 8889 is listening");
        return;
    }
    let dir = model_dir();
    if !Path::new(&dir).is_dir() {
        eprintln!("skipping prefill_lin: {dir} is missing");
        return;
    }

    let _cache = CacheGuard::snapshot();
    std::env::set_var("EXL3_HGEMM_F16ACC", "auto");
    std::env::set_var("EXL3_INT8_GEMV", "2");
    std::env::set_var("EXL3_INT8_GEMV_MAX_K", "5");
    std::env::set_var("EXL3_QC_STAGING", "1");

    let ckpt = Checkpoint::open(&dir).unwrap_or_else(|err| panic!("open {dir}: {err}"));
    let cases = specs();
    // Several GB of generated references: on disk, not the /tmp tmpfs.
    // One fixed directory, emptied first, so repeated runs do not pile up.
    let out_dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("berth-prefill-lin");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();

    let mut inputs = Vec::new();
    let mut manifest_cases = Vec::new();
    for spec in &cases {
        let item = linear(&ckpt, &spec.prefixes[0]);
        let (k_in, _) = shape_kn(item);
        let mut seed = 0xC0FFEE_u64;
        for byte in spec.id.bytes() {
            seed = seed.wrapping_mul(131).wrapping_add(u64::from(byte));
        }
        let input = fill_f16(seed, spec.rows as usize * k_in as usize);
        let input_name = format!("{}.in", spec.id);
        std::fs::write(out_dir.join(&input_name), &input).unwrap();
        let mut case = serde_json::json!({
            "id": spec.id,
            "kind": spec.kind,
            "rows": spec.rows,
            "out": match spec.out {
                OutDtype::F16 => "f16",
                OutDtype::F32 => "f32",
            },
            "input": input_name,
        });
        if spec.kind == "linear" {
            case["prefix"] = serde_json::json!(spec.prefixes[0]);
        } else {
            case["prefixes"] = serde_json::json!(spec.prefixes);
        }
        manifest_cases.push(case);
        inputs.push(input);
    }
    let manifest_path = out_dir.join("manifest.json");
    let manifest = serde_json::json!({
        "model": dir,
        "out_dir": out_dir,
        "cases": manifest_cases,
    });
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    // The fork writes the coop autotune file on a miss. Run it before the
    // shim so both processes read the same records.
    run_python(&manifest_path);

    let ctx = Context::new().unwrap();
    let stream = ctx.stream().unwrap();
    for (spec, input) in cases.iter().zip(inputs.iter()) {
        if spec.kind == "linear" {
            run_linear_case(&stream, &ckpt, spec, input, &out_dir);
        } else {
            run_mgemm_case(&stream, &ckpt, spec, input, &out_dir);
        }
    }
}
