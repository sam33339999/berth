//! Gated DeltaNet prefill against the fork, one chunk, batch 1, history off.
//!
//! `gdn_prefill_ref.py` runs the non-BC path (`gated_delta_net.py` 1096–1151)
//! on the cases in `testdata/kernels/gdn_prefill/meta.json` and writes the
//! outputs under `CARGO_TARGET_TMPDIR` (not `/tmp`). The GPU test runs only
//! with `BERTH_GPU_TESTS=1` and skips when `:8889` is listening. It does not
//! load the 27B: the reference reads the four layer-0 tensors and writes them
//! next to the outputs.
//!
//! `history_stride` is `recurrent_state.size(1)`, which
//! `cuda_recurrent_gated_delta_rule_gr` passes through (`gdn.cu` 933). The
//! kernel indexes `state_slot * history_stride` (`gdn.cu` 691–699) and, with
//! history off, writes history index 0 in place. The chunk path has no slots
//! argument; `gated_delta_rule.py` 180–190 views `recurrent_state[slot, 0]`
//! and copies the final state back.

use std::path::{Path, PathBuf};
use std::process::Command;

use berth_cuda::{Context, DeviceBuffer};
use berth_kernels::gdn_prefill::{scratch_bytes, GdnPrefill, GdnPrefillArgs};

const PY: &str = "/home/sam/qwen38-exl3-dflash2/env/bin/python";
const PYTHONPATH: &str = "/home/sam/qwen38-exl3-dflash2/exllamav3";
const QKV_DIM: usize = 10240;
const V_DIM: usize = 6144;
const NUM_V_HEADS: usize = 48;
const HEAD_DIM: usize = 128;
const CONV_K: usize = 4;

fn meta_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/kernels/gdn_prefill/meta.json")
}

fn cubin_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins/fla")
}

fn load_meta() -> serde_json::Value {
    let path = meta_path();
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()))
}

fn lcg_f32(seed: u32, n: usize) -> Vec<u8> {
    let mut x = seed;
    let mut out = Vec::with_capacity(n * 4);
    for _ in 0..n {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let mut bits = (x & 0x007f_ffff) | 0x3f00_0000;
        if x & 0x100 != 0 {
            bits |= 0x8000_0000;
        }
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

/// High 16 bits of the same LCG stream. Truncation, not round-to-nearest.
fn lcg_bf16_trunc(seed: u32, n: usize) -> Vec<u8> {
    let mut x = seed;
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let mut bits = (x & 0x007f_ffff) | 0x3f00_0000;
        if x & 0x100 != 0 {
            bits |= 0x8000_0000;
        }
        let hi = (bits >> 16) as u16;
        out.extend_from_slice(&hi.to_le_bytes());
    }
    out
}

fn serve_blocks_gpu() -> bool {
    let output = match Command::new("ss").args(["-ltn"]).output() {
        Ok(output) => output,
        Err(_) => return true,
    };
    if !output.status.success() {
        return true;
    }
    String::from_utf8_lossy(&output.stdout).contains(":8889 ")
}

fn assert_same(name: &str, got: &[u8], want: &[u8], width: usize) {
    assert_eq!(
        got.len(),
        want.len(),
        "{name}: {} bytes != {}",
        got.len(),
        want.len()
    );
    assert!(
        got.len().is_multiple_of(width),
        "{name}: length is not a multiple of {width}"
    );
    let mut differing = 0usize;
    let mut first = None;
    for (index, (g, w)) in got.chunks(width).zip(want.chunks(width)).enumerate() {
        if g != w {
            if first.is_none() {
                first = Some(index);
            }
            differing += 1;
        }
    }
    if differing != 0 {
        let index = first.unwrap();
        let start = index * width;
        panic!(
            "{name}: {differing} of {} elements differ; first at {index}, got {:02x?} want {:02x?}",
            got.len() / width,
            &got[start..start + width],
            &want[start..start + width],
        );
    }
}

fn read_exact(path: &Path, len: usize) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    assert_eq!(
        bytes.len(),
        len,
        "{} is {} bytes, want {len}",
        path.display(),
        bytes.len()
    );
    bytes
}

/// Torch's bundled CUDA libs, the same prefix `scripts/serve-host.sh` exports.
fn fork_ld_library_path() -> String {
    let script = "\
import pathlib, torch
base = pathlib.Path(torch.__file__).resolve().parent.parent / 'nvidia'
libs = sorted(p for p in base.glob('*/lib') if p.is_dir())
print(':'.join(str(p) for p in libs))
";
    let out = Command::new(PY)
        .arg("-c")
        .arg(script)
        .output()
        .unwrap_or_else(|err| panic!("spawn python for LD_LIBRARY_PATH: {err}"));
    assert!(
        out.status.success(),
        "LD_LIBRARY_PATH probe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let nvidia = String::from_utf8(out.stdout).unwrap();
    let nvidia = nvidia.trim();
    assert!(!nvidia.is_empty(), "torch nvidia/*/lib is empty");
    match std::env::var("LD_LIBRARY_PATH") {
        Ok(rest) if !rest.is_empty() => format!("{nvidia}:{rest}"),
        _ => nvidia.to_string(),
    }
}

fn has_case(meta: &serde_json::Value, seqlen: i64, zero_state: bool) -> bool {
    meta["cases"].as_array().unwrap().iter().any(|case| {
        case["seqlen"].as_i64() == Some(seqlen)
            && case["zero_state"].as_bool() == Some(zero_state)
            && case["slot"].as_i64() == Some(0)
            && case["num_slots"].as_i64() == Some(1)
            && case["history_stride"].as_i64() == Some(1)
    })
}

#[test]
fn meta_lists_the_prefill_cases() {
    let meta = load_meta();
    assert_eq!(meta["prefix_bytes"].as_u64().unwrap(), 32);
    assert_eq!(meta["beta_scale"].as_f64().unwrap(), 1.0);
    for seqlen in [47, 300, 1499] {
        assert!(
            has_case(&meta, seqlen, true),
            "missing zero state seq {seqlen}"
        );
        assert!(
            has_case(&meta, seqlen, false),
            "missing nonzero state seq {seqlen}"
        );
    }
}

#[test]
fn gdn_prefill_matches_the_fork() {
    if std::env::var("BERTH_GPU_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping gdn prefill gpu test: set BERTH_GPU_TESTS=1");
        return;
    }
    if serve_blocks_gpu() {
        eprintln!("skipping gdn prefill gpu test: port 8889 is listening or ss failed");
        return;
    }

    let meta = load_meta();
    let cases = meta["cases"].as_array().unwrap();
    assert!(!cases.is_empty());
    let prefix_bytes = meta["prefix_bytes"].as_u64().unwrap() as usize;
    assert_eq!(prefix_bytes, 32);
    let beta_scale = meta["beta_scale"].as_f64().unwrap() as f32;
    let seeds = &meta["seeds"];
    let seed = |key: &str| seeds[key].as_u64().unwrap() as u32;

    let out_dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("gdn_prefill_ref");
    if out_dir.exists() {
        std::fs::remove_dir_all(&out_dir).unwrap();
    }
    std::fs::create_dir_all(&out_dir).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/gdn_prefill_ref.py");
    let manifest = cubin_dir().join("manifest.json");
    assert!(
        manifest.is_file(),
        "missing {}. Run scripts/build_triton_fla_cubins.py",
        manifest.display()
    );
    let reference = Command::new(PY)
        .arg(&script)
        .arg("--out")
        .arg(&out_dir)
        .arg("--manifest")
        .arg(&manifest)
        .arg("--meta")
        .arg(meta_path())
        .env("PYTHONPATH", PYTHONPATH)
        .env("PYTHONUNBUFFERED", "1")
        .env("TRITON_LIBCUDA_PATH", "/usr/lib/x86_64-linux-gnu")
        .env("LD_LIBRARY_PATH", fork_ld_library_path())
        .env("EXL3_INT8_GEMV", "2")
        .env("EXL3_INT8_GEMV_MAX_K", "5")
        .env("EXL3_QC_STAGING", "1")
        .env("EXL3_HGEMM_F16ACC", "auto")
        .output()
        .unwrap_or_else(|err| panic!("spawn reference: {err}"));
    assert!(
        reference.status.success(),
        "gdn_prefill_ref.py failed ({})\nstdout:\n{}\nstderr:\n{}",
        reference.status,
        String::from_utf8_lossy(&reference.stdout),
        String::from_utf8_lossy(&reference.stderr)
    );
    eprintln!("{}", String::from_utf8_lossy(&reference.stdout));

    let ctx = Context::new().unwrap();
    let stream = ctx.stream().unwrap();
    let weights = out_dir.join("weights");
    let a_log = read_exact(&weights.join("a_log.bin"), NUM_V_HEADS * 2);
    let dt_bias = read_exact(&weights.join("dt_bias.bin"), NUM_V_HEADS * 2);
    let conv_weight = read_exact(&weights.join("conv_weight.bin"), QKV_DIM * CONV_K * 2);
    let norm_weight = read_exact(&weights.join("norm_weight.bin"), HEAD_DIM * 2);
    let a_log_d = DeviceBuffer::from_host(&stream, &a_log).unwrap();
    let dt_bias_d = DeviceBuffer::from_host(&stream, &dt_bias).unwrap();
    let conv_weight_d = DeviceBuffer::from_host(&stream, &conv_weight).unwrap();
    let norm_weight_d = DeviceBuffer::from_host(&stream, &norm_weight).unwrap();
    let prefill = GdnPrefill::load(&ctx, &cubin_dir()).unwrap();

    for (index, case) in cases.iter().enumerate() {
        let seqlen = case["seqlen"].as_i64().unwrap() as i32;
        let zero = case["zero_state"].as_bool().unwrap();
        let slot = case["slot"].as_i64().unwrap() as i32;
        let num_slots = case["num_slots"].as_i64().unwrap() as i32;
        let history_stride = case["history_stride"].as_i64().unwrap() as i32;
        let label =
            format!("seq={seqlen} zero={zero} slot={slot} slots={num_slots} hs={history_stride}");
        let s = seqlen as usize;
        let conv_n = num_slots as usize * QKV_DIM * CONV_K;
        let rec_n =
            num_slots as usize * history_stride as usize * NUM_V_HEADS * HEAD_DIM * HEAD_DIM;
        let qkv = lcg_f32(seed("qkv"), s * QKV_DIM);
        let z = lcg_f32(seed("z"), s * V_DIM);
        let b = lcg_f32(seed("b"), s * NUM_V_HEADS);
        let a = lcg_f32(seed("a"), s * NUM_V_HEADS);
        let conv = if zero {
            vec![0u8; conv_n * 2]
        } else {
            lcg_bf16_trunc(seed("conv"), conv_n)
        };
        let rec = if zero {
            vec![0u8; rec_n * 4]
        } else {
            lcg_f32(seed("rec"), rec_n)
        };

        let case_dir = out_dir.join(format!("case_{index:02}"));
        let check_prefix = |name: &str, data: &[u8]| {
            let path = case_dir.join(name);
            let prefix = read_exact(&path, prefix_bytes);
            assert_eq!(&data[..prefix_bytes], prefix.as_slice(), "{label} {name}");
        };
        check_prefix("qkv_prefix.bin", &qkv);
        check_prefix("z_prefix.bin", &z);
        check_prefix("b_prefix.bin", &b);
        check_prefix("a_prefix.bin", &a);
        check_prefix("conv_prefix.bin", &conv);
        check_prefix("rec_prefix.bin", &rec);

        let workspace_bytes = scratch_bytes(seqlen).unwrap();
        let qkv_d = DeviceBuffer::from_host(&stream, &qkv).unwrap();
        let z_d = DeviceBuffer::from_host(&stream, &z).unwrap();
        let b_d = DeviceBuffer::from_host(&stream, &b).unwrap();
        let a_d = DeviceBuffer::from_host(&stream, &a).unwrap();
        let conv_d = DeviceBuffer::from_host(&stream, &conv).unwrap();
        let rec_d = DeviceBuffer::from_host(&stream, &rec).unwrap();
        let out_bytes = s * V_DIM * 2;
        let out_d = DeviceBuffer::<u8>::zeroed(&stream, out_bytes).unwrap();
        let workspace = DeviceBuffer::<u8>::zeroed(&stream, workspace_bytes).unwrap();
        assert!(
            workspace.ptr().as_u64().is_multiple_of(256),
            "cuda malloc is not 256-byte aligned"
        );

        let args = GdnPrefillArgs {
            qkv: qkv_d.ptr().as_u64(),
            z: z_d.ptr().as_u64(),
            b: b_d.ptr().as_u64(),
            a: a_d.ptr().as_u64(),
            dt_bias: dt_bias_d.ptr().as_u64(),
            a_log: a_log_d.ptr().as_u64(),
            conv_weight: conv_weight_d.ptr().as_u64(),
            norm_weight: norm_weight_d.ptr().as_u64(),
            conv_state: conv_d.ptr().as_u64(),
            recurrent_state: rec_d.ptr().as_u64(),
            out: out_d.ptr().as_u64(),
            workspace: workspace.ptr().as_u64(),
            workspace_bytes,
            seqlen,
            slot,
            num_slots,
            history_stride,
            beta_scale,
        };
        // SAFETY: each buffer is a live device allocation of the stated shape
        // on `stream`. copy_to_host below synchronizes that stream before the
        // buffers are read or dropped.
        unsafe { prefill.run(&stream, &args) }.unwrap_or_else(|err| panic!("{label}: {err}"));

        let mut got_out = vec![0u8; out_bytes];
        let mut got_conv = vec![0u8; conv.len()];
        let mut got_rec = vec![0u8; rec.len()];
        out_d.copy_to_host(&stream, &mut got_out).unwrap();
        conv_d.copy_to_host(&stream, &mut got_conv).unwrap();
        rec_d.copy_to_host(&stream, &mut got_rec).unwrap();

        let want_out = read_exact(&case_dir.join("out.bin"), out_bytes);
        let want_conv = read_exact(&case_dir.join("conv_state.bin"), conv.len());
        let want_rec = read_exact(&case_dir.join("recurrent.bin"), rec.len());
        assert_same(&format!("{label} core"), &got_out, &want_out, 2);
        assert_same(&format!("{label} conv state"), &got_conv, &want_conv, 2);
        assert_same(&format!("{label} recurrent state"), &got_rec, &want_rec, 4);
        eprintln!("{label}: 0 differing");
    }
}
