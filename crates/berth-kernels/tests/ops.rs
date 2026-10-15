//! Decode ops, bit for bit against the fork reference in `testdata/kernels/ops`.
//!
//! `tests/ops_ref.py` writes the activations and the fork outputs. Trellis,
//! Hadamard scales, norm weights, and embedding rows are read here from the
//! 27B checkpoint with `berth-exl3`. The test skips unless `BERTH_GPU_TESTS=1`,
//! and also skips when the checkpoint is missing or port 8889 is listening.
//!
//! The mgemm launches reuse the service autotune cache. This test refuses to
//! launch unless the winning records are the measured shape-4 geometry, and
//! restores the cache file if a launch appends to it.

use std::ffi::c_void;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use berth_cuda::{Context, DeviceBuffer, Stream};
use berth_exl3::{Bitrate, Checkpoint, HadamardScale, LinearExl3, TensorInfo};
use berth_kernels::ops::{self, NormWeight};
use berth_kernels::{Codebook, OutDtype};

const GDN_KEY: u64 = 0xd229_7a8a_33a3_3f5b;
const ATTN_KEY: u64 = 0x9c25_a075_2178_fb4a;
const CACHE_TAG: i32 = 4;
const CACHE_BLOCK: i32 = 256;
const CACHE_SMS: i32 = 10;
const CACHE_CONCURRENCY: i32 = 8;

fn model_dir() -> String {
    std::env::var("BERTH_27B_DIR")
        .unwrap_or_else(|_| "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3".to_string())
}

fn ref_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/kernels/ops")
}

fn serve_listening() -> bool {
    let output = match Command::new("ss").args(["-ltn"]).output() {
        Ok(output) => output,
        Err(_) => return true,
    };
    if !output.status.success() {
        return true;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.contains(":8889 ")
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

fn salt(hash: u64) -> u64 {
    (hash ^ 4).wrapping_mul(1_099_511_628_211)
}

struct CacheRecord {
    tag: i32,
    block_dim: i32,
    num_sms: i32,
    concurrency: i32,
}

fn last_record(bytes: &[u8], hash: u64) -> Option<CacheRecord> {
    if bytes.len() < 16 || &bytes[..8] != b"EX3ATUNE" {
        return None;
    }
    let format = u32::from_le_bytes(bytes[8..12].try_into().ok()?);
    let record_size = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    if format != 1 || record_size < 32 {
        return None;
    }
    let mut found = None;
    let mut off = 16;
    while off + record_size <= bytes.len() {
        let chunk = &bytes[off..off + record_size];
        if &chunk[..8] == b"EX3ATUNE" {
            off += 16;
            continue;
        }
        let record_hash = u64::from_le_bytes(chunk[0..8].try_into().ok()?);
        if record_hash == hash {
            found = Some(CacheRecord {
                tag: i32::from_le_bytes(chunk[8..12].try_into().ok()?),
                block_dim: i32::from_le_bytes(chunk[12..16].try_into().ok()?),
                num_sms: i32::from_le_bytes(chunk[16..20].try_into().ok()?),
                concurrency: i32::from_le_bytes(chunk[20..24].try_into().ok()?),
            });
        }
        off += record_size;
    }
    found
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

    fn changed(&self) -> bool {
        std::fs::read(&self.path).ok().as_deref() != Some(self.original.as_slice())
    }
}

impl Drop for CacheGuard {
    fn drop(&mut self) {
        if self.changed() {
            if let Err(err) = std::fs::write(&self.path, &self.original) {
                eprintln!("failed to restore {}: {err}", self.path.display());
            } else {
                eprintln!("restored {}", self.path.display());
            }
        }
    }
}

fn read_file(path: &Path) -> Vec<u8> {
    std::fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
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

fn read_row(info: &TensorInfo, row: u64, row_bytes: u64) -> Vec<u8> {
    let mut file =
        File::open(&info.shard).unwrap_or_else(|err| panic!("{}: {err}", info.shard.display()));
    file.seek(SeekFrom::Start(info.offset + row * row_bytes))
        .unwrap_or_else(|err| panic!("seek {}: {err}", info.name));
    let mut bytes = vec![0u8; row_bytes as usize];
    file.read_exact(&mut bytes)
        .unwrap_or_else(|err| panic!("read {}: {err}", info.name));
    bytes
}

fn upload(stream: &Stream, bytes: &[u8]) -> DeviceBuffer<u8> {
    DeviceBuffer::from_host(stream, bytes).unwrap()
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
        .find(|linear| linear.prefix == prefix)
        .unwrap_or_else(|| panic!("{prefix} is not an EXL3 linear"))
}

fn differ(got: &[u8], want: &[u8], elem: usize) -> usize {
    assert_eq!(got.len(), want.len());
    got.chunks(elem)
        .zip(want.chunks(elem))
        .filter(|(g, w)| g != w)
        .count()
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}

fn u64s(stream: &Stream, values: &[u64]) -> DeviceBuffer<u64> {
    DeviceBuffer::from_host(stream, values).unwrap()
}

fn i32s(stream: &Stream, values: &[i32]) -> DeviceBuffer<i32> {
    DeviceBuffer::from_host(stream, values).unwrap()
}

struct Source {
    n: i32,
    trellis: DeviceBuffer<u8>,
    suh: DeviceBuffer<u8>,
    svh: DeviceBuffer<u8>,
    out: DeviceBuffer<u8>,
}

fn load_source(stream: &Stream, linear: &LinearExl3, elem: usize) -> Source {
    let Bitrate::Int(k_bits) = linear.k else {
        panic!("{}: half-integer rate", linear.prefix);
    };
    assert_eq!(k_bits, 4, "{}", linear.prefix);
    assert!(
        matches!(linear.codebook, berth_exl3::Codebook::Mul1),
        "{}",
        linear.prefix
    );
    let shape = &linear.trellis.shape;
    assert_eq!(shape.len(), 3, "{}", linear.prefix);
    let n = i32::try_from(shape[1] * 16).unwrap();
    let k_in = shape[0] * 16;
    assert_eq!(k_in, 5120, "{}", linear.prefix);
    assert_eq!(shape[2] / 16, 4, "{}", linear.prefix);
    let suh = unpacked(&linear.input_scale);
    let svh = unpacked(&linear.output_scale);
    assert_eq!(suh.dtype, berth_exl3::DType::F16);
    assert_eq!(svh.dtype, berth_exl3::DType::F16);
    assert_eq!(suh.length, k_in * 2);
    assert_eq!(svh.length, n as u64 * 2);
    Source {
        n,
        trellis: upload(stream, &read_tensor(&linear.trellis)),
        suh: upload(stream, &read_tensor(suh)),
        svh: upload(stream, &read_tensor(svh)),
        out: DeviceBuffer::zeroed(stream, n as usize * elem).unwrap(),
    }
}

fn run_mgemm(
    stream: &Stream,
    ckpt: &Checkpoint,
    case: &serde_json::Value,
    x: &[u8],
) -> (Vec<Vec<u8>>, String) {
    let prefixes: Vec<&str> = case["prefixes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let out_elems: Vec<i32> = case["out_elems"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect();
    let out_dtype = case["out_dtype"].as_str().unwrap();
    let (dtype, elem) = match out_dtype {
        "float32" => (OutDtype::F32, 4),
        "float16" => (OutDtype::F16, 2),
        other => panic!("output dtype {other}"),
    };
    let sources: Vec<Source> = prefixes
        .iter()
        .zip(&out_elems)
        .map(|(prefix, n)| {
            let source = load_source(stream, linear(ckpt, prefix), elem);
            assert_eq!(source.n, *n, "{prefix}");
            source
        })
        .collect();
    let width = sources.iter().fold(0, |acc, source| {
        if acc == 0 {
            source.n
        } else {
            gcd(acc, source.n)
        }
    });
    assert_eq!(width, case["width"].as_i64().unwrap() as i32);
    assert!(width % 128 == 0 && width >= 256);

    let mut b_host = Vec::new();
    let mut svh_host = Vec::new();
    let mut suh_host = Vec::new();
    let mut c_host = Vec::new();
    let mut size_n = Vec::new();
    let mut strides = Vec::new();
    let mut had_src = Vec::new();
    for (index, source) in sources.iter().enumerate() {
        suh_host.push(source.suh.ptr().as_u64());
        let mut n0 = 0;
        while n0 < source.n {
            let trellis_off = (n0 / 16) * 16 * 4 * 2;
            b_host.push(source.trellis.ptr().as_u64() + trellis_off as u64);
            svh_host.push(source.svh.ptr().as_u64() + (n0 as u64) * 2);
            c_host.push(source.out.ptr().as_u64() + (n0 as u64) * (elem as u64));
            size_n.push(width);
            strides.push(source.n);
            had_src.push(index as i32);
            n0 += width;
        }
    }
    let slices = had_src.len() as i32;
    assert_eq!(slices, case["slices"].as_i64().unwrap() as i32);
    assert_eq!(suh_host.len(), case["sources"].as_u64().unwrap() as usize);

    let b_list = u64s(stream, &b_host);
    let suh_list = u64s(stream, &suh_host);
    let svh_list = u64s(stream, &svh_host);
    let c_ptrs = u64s(stream, &c_host);
    let size_n_list = i32s(stream, &size_n);
    let n_stride_list = i32s(stream, &strides);
    let had_src_list = i32s(stream, &had_src);
    let a = upload(stream, x);
    let a_had = DeviceBuffer::<u8>::zeroed(stream, suh_host.len() * x.len()).unwrap();
    let carrier =
        DeviceBuffer::<u8>::zeroed(stream, slices as usize * width as usize * elem).unwrap();

    let hidden = x.len() as i32 / 2;
    // SAFETY: the buffers above are live device allocations for this sliced
    // launch, and copy_to_host below synchronizes the stream before they drop.
    let tag = unsafe {
        ops::exl3_mgemm(
            a.ptr().as_u64() as *const c_void,
            b_list.ptr().as_u64() as *const c_void,
            carrier.ptr().as_u64() as *mut c_void,
            suh_list.ptr().as_u64() as *const c_void,
            a_had.ptr().as_u64() as *mut c_void,
            svh_list.ptr().as_u64() as *const c_void,
            1,
            hidden,
            width,
            4,
            Codebook::Mul1,
            dtype,
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
    .unwrap_or_else(|err| panic!("{}: {err}", case["name"]));
    assert_eq!(tag, 4, "{}", case["name"]);
    let kernel = ops::last_launch();
    let mut outputs = Vec::new();
    for source in &sources {
        let mut got = vec![0u8; source.n as usize * elem];
        source.out.copy_to_host(stream, &mut got).unwrap();
        outputs.push(got);
    }
    (outputs, kernel)
}

fn dev_ptr(buf: &DeviceBuffer<u8>) -> *mut c_void {
    buf.ptr().as_u64() as *mut c_void
}

#[test]
fn ops_match_the_fork_bit_for_bit() {
    if std::env::var("BERTH_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping ops: set BERTH_GPU_TESTS=1");
        return;
    }
    if serve_listening() {
        eprintln!("skipping ops: port 8889 is listening");
        return;
    }
    let dir = model_dir();
    if !Path::new(&dir).is_dir() {
        eprintln!("skipping ops: directory missing: {dir}");
        return;
    }
    let data = ref_dir();
    let meta: serde_json::Value =
        serde_json::from_slice(&read_file(&data.join("meta.json"))).unwrap();
    let hidden = meta["hidden"].as_u64().unwrap() as i32;
    assert_eq!(hidden, 5120);

    let ctx = Context::new().unwrap();
    let sms = ctx.multiprocessor_count().unwrap();
    let cc = ctx.compute_capability().unwrap();
    assert_eq!(
        sms, 82,
        "mgemm cache is for the 82-SM device; refusing to retune"
    );
    assert_eq!(
        (cc.major, cc.minor),
        (8, 6),
        "mgemm cache is for sm_86; refusing to retune"
    );

    let cache = CacheGuard::snapshot();
    for (label, key, n, c_fp32, bszm_out) in [
        ("gdn", GDN_KEY, 2048, 1, 8),
        ("attn", ATTN_KEY, 1024, 0, 14),
    ] {
        let got = ops::mgemm_tune_key(
            1,
            hidden,
            n,
            4,
            if c_fp32 == 1 {
                OutDtype::F32
            } else {
                OutDtype::F16
            },
            0,
            2,
            82,
            Codebook::Mul1,
            1,
            bszm_out,
            true,
        );
        assert_eq!(got, key, "{label}");
        let record = last_record(&cache.original, salt(key))
            .unwrap_or_else(|| panic!("{label}: tune record missing"));
        assert_eq!(
            (
                record.tag,
                record.block_dim,
                record.num_sms,
                record.concurrency
            ),
            (CACHE_TAG, CACHE_BLOCK, CACHE_SMS, CACHE_CONCURRENCY),
            "{label}: refusing to launch a cache miss"
        );
    }

    let ckpt = Checkpoint::open(&dir).unwrap();
    let stream = ctx.stream().unwrap();
    let x = read_file(&data.join(meta["x_file"].as_str().unwrap()));
    assert_eq!(x.len(), hidden as usize * 2);

    let mut failures = Vec::new();
    for case in meta["mgemm"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let elem = if case["out_dtype"].as_str().unwrap() == "float32" {
            4
        } else {
            2
        };
        let (outputs, kernel) = run_mgemm(&stream, &ckpt, case, &x);
        assert_eq!(kernel, case["kernel"].as_str().unwrap(), "{name}");
        let files = case["out_files"].as_array().unwrap();
        let elems = case["out_elems"].as_array().unwrap();
        for ((got, file), n) in outputs.iter().zip(files).zip(elems) {
            let want = read_file(&data.join(file.as_str().unwrap()));
            let n = n.as_u64().unwrap() as usize;
            let differing = differ(got, &want, elem);
            eprintln!(
                "{name} {}: {kernel}, {differing} of {n} elements differ",
                file.as_str().unwrap()
            );
            if differing != 0 {
                failures.push(format!("{name}: {differing} of {n} elements differ"));
            }
        }
    }
    if cache.changed() {
        panic!("mgemm wrote the autotune cache; the guard restores it on drop");
    }

    let eps = f32::from_bits(meta["epsilon_f32_bits"].as_u64().unwrap() as u32);
    let bias = meta["constant_bias"].as_f64().unwrap() as f32;
    let scale = meta["constant_scale"].as_f64().unwrap() as f32;
    for case in meta["norms"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let weight_name = case["weight"].as_str().unwrap();
        let weight_info = ckpt
            .tensors
            .get(weight_name)
            .unwrap_or_else(|| panic!("{weight_name}"));
        assert_eq!(weight_info.dtype, berth_exl3::DType::Bf16);
        assert_eq!(weight_info.shape, vec![hidden as u64]);
        let weight = upload(&stream, &read_tensor(weight_info));
        let x_bytes = read_file(&data.join(case["x"].as_str().unwrap()));
        let x_buf = upload(&stream, &x_bytes);
        let y_buf = DeviceBuffer::<u8>::zeroed(&stream, hidden as usize * 2).unwrap();
        let kind = case["kind"].as_str().unwrap();
        if kind == "none" {
            // SAFETY: x, weight, and y are live device buffers of this row.
            unsafe {
                ops::rms_norm(
                    x_buf.ptr().as_u64() as *const c_void,
                    weight.ptr().as_u64() as *const c_void,
                    dev_ptr(&y_buf),
                    1,
                    hidden,
                    eps,
                    bias,
                    scale,
                    NormWeight::Bf16,
                    stream.handle().as_ptr(),
                )
            }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        } else {
            let r_bytes = read_file(&data.join(case["residual"].as_str().unwrap()));
            let r_buf = upload(&stream, &r_bytes);
            // SAFETY: x, weight, y, and residual are live. residual is overwritten.
            unsafe {
                ops::rms_norm_res_in(
                    x_buf.ptr().as_u64() as *const c_void,
                    weight.ptr().as_u64() as *const c_void,
                    dev_ptr(&y_buf),
                    dev_ptr(&r_buf),
                    1,
                    hidden,
                    eps,
                    bias,
                    scale,
                    NormWeight::Bf16,
                    stream.handle().as_ptr(),
                )
            }
            .unwrap_or_else(|err| panic!("{name}: {err}"));
            let mut got_r = vec![0u8; r_bytes.len()];
            r_buf.copy_to_host(&stream, &mut got_r).unwrap();
            let want_r = read_file(&data.join(case["residual_out"].as_str().unwrap()));
            let differing = differ(&got_r, &want_r, 4);
            eprintln!("{name} residual: {differing} of {hidden} elements differ");
            if differing != 0 {
                failures.push(format!("{name} residual: {differing} differ"));
            }
        }
        let kernel = ops::last_launch();
        assert_eq!(kernel, case["kernel"].as_str().unwrap(), "{name}");
        let mut got = vec![0u8; hidden as usize * 2];
        y_buf.copy_to_host(&stream, &mut got).unwrap();
        let want = read_file(&data.join(case["y"].as_str().unwrap()));
        let differing = differ(&got, &want, 2);
        eprintln!("{name}: {kernel}, {differing} of {hidden} elements differ");
        if differing != 0 {
            failures.push(format!("{name}: {differing} of {hidden} elements differ"));
        }
    }

    let silu = &meta["silu"];
    let numel = silu["numel"].as_u64().unwrap();
    let gate = upload(
        &stream,
        &read_file(&data.join(silu["gate"].as_str().unwrap())),
    );
    let up = upload(
        &stream,
        &read_file(&data.join(silu["up"].as_str().unwrap())),
    );
    let activated = DeviceBuffer::<u8>::zeroed(&stream, numel as usize * 2).unwrap();
    let act_limit = silu["act_limit"].as_f64().unwrap() as f32;
    // SAFETY: gate, up, and out are live fp16 buffers of `numel` elements.
    unsafe {
        ops::silu_mul(
            gate.ptr().as_u64() as *const c_void,
            up.ptr().as_u64() as *const c_void,
            dev_ptr(&activated),
            numel as i64,
            act_limit,
            stream.handle().as_ptr(),
        )
    }
    .unwrap_or_else(|err| panic!("silu: {err}"));
    let kernel = ops::last_launch();
    assert_eq!(kernel, silu["kernel"].as_str().unwrap());
    let mut got = vec![0u8; numel as usize * 2];
    activated.copy_to_host(&stream, &mut got).unwrap();
    let want = read_file(&data.join(silu["out"].as_str().unwrap()));
    let differing = differ(&got, &want, 2);
    eprintln!("silu: {kernel}, {differing} of {numel} elements differ");
    if differing != 0 {
        failures.push(format!("silu: {differing} of {numel} elements differ"));
    }

    let add = &meta["add"];
    let n = add["n"].as_u64().unwrap() as i32;
    let a = upload(&stream, &read_file(&data.join(add["a"].as_str().unwrap())));
    let b = upload(&stream, &read_file(&data.join(add["b"].as_str().unwrap())));
    let c = DeviceBuffer::<u8>::zeroed(&stream, n as usize * 4).unwrap();
    // SAFETY: a, b, and c are live fp32 buffers of `n` elements. c does not alias a.
    unsafe {
        ops::add_f32(
            a.ptr().as_u64() as *const c_void,
            b.ptr().as_u64() as *const c_void,
            dev_ptr(&c),
            n,
            stream.handle().as_ptr(),
        )
    }
    .unwrap_or_else(|err| panic!("add: {err}"));
    let kernel = ops::last_launch();
    assert_eq!(kernel, add["kernel"].as_str().unwrap());
    let mut got = vec![0u8; n as usize * 4];
    c.copy_to_host(&stream, &mut got).unwrap();
    let want = read_file(&data.join(add["c"].as_str().unwrap()));
    let differing = differ(&got, &want, 4);
    eprintln!("add: {kernel}, {differing} of {n} elements differ");
    if differing != 0 {
        failures.push(format!("add: {differing} of {n} elements differ"));
    }

    let embed = &meta["embed"];
    let token_ids: Vec<u64> = embed["token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let weight_name = embed["weight"].as_str().unwrap();
    let weight_info = ckpt
        .tensors
        .get(weight_name)
        .unwrap_or_else(|| panic!("{weight_name}"));
    assert_eq!(weight_info.dtype, berth_exl3::DType::Bf16);
    assert_eq!(
        weight_info.shape,
        vec![meta["vocab"].as_u64().unwrap(), hidden as u64]
    );
    let mut table = Vec::new();
    for id in &token_ids {
        table.extend(read_row(weight_info, *id, hidden as u64 * 2));
    }
    let ids = read_file(&data.join(embed["local_ids"].as_str().unwrap()));
    assert_eq!(ids.len(), token_ids.len() * 4);
    let weight = upload(&stream, &table);
    let id_buf = upload(&stream, &ids);
    let embedded =
        DeviceBuffer::<u8>::zeroed(&stream, token_ids.len() * hidden as usize * 4).unwrap();
    // SAFETY: `weight` holds one bf16 row per table entry, `ids` are i32
    // indexes into that table, and `embedded` is fp32 [n, hidden].
    unsafe {
        ops::embed_tokens(
            weight.ptr().as_u64() as *const c_void,
            id_buf.ptr().as_u64() as *const i32,
            dev_ptr(&embedded),
            hidden,
            token_ids.len() as i32,
            stream.handle().as_ptr(),
        )
    }
    .unwrap_or_else(|err| panic!("embed: {err}"));
    let kernel = ops::last_launch();
    assert_eq!(kernel, embed["kernel"].as_str().unwrap());
    let mut got = vec![0u8; token_ids.len() * hidden as usize * 4];
    embedded.copy_to_host(&stream, &mut got).unwrap();
    let want = read_file(&data.join(embed["out"].as_str().unwrap()));
    let differing = differ(&got, &want, 4);
    let n_out = token_ids.len() * hidden as usize;
    eprintln!("embed: {kernel}, {differing} of {n_out} elements differ");
    if differing != 0 {
        failures.push(format!("embed: {differing} of {n_out} elements differ"));
    }

    assert!(failures.is_empty(), "{failures:#?}");
}
