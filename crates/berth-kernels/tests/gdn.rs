//! Gated DeltaNet decode kernels against the fork, one token, 27B shapes.
//!
//! `gdn_ref.py` writes `testdata/kernels/gdn/`. The GPU test runs only with
//! `BERTH_GPU_TESTS=1` and skips when `:8889` is listening. It does not load
//! the 27B. The recurrent state is not stored (3 MiB would blow the testdata
//! budget): both sides expand it with the LCG in `meta.json`, and the updated
//! state matches when its sha256 matches. That hash is over every little-endian
//! f32 byte, so a match means zero differing elements. `core_attn_out` is also
//! compared element-wise.
//!
//! `gated_delta_net_fused_op_3` has no Python binding. The reference builds
//! `mixed_qkv` with a transpose and bf16 cast and builds beta/g with
//! `gated_delta_net_fused_op_2`. See `meta.json` field `fused_op_3`.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use berth_cuda::{Context, DeviceBuffer};
use berth_kernels::gdn;

const LAUNCH_BA: &str = "gdn_ba_gemv_kernel grid=12x1x1 block=256x1x1 smem=0";
const LAUNCH_FUSED: &str =
    "gated_delta_net_fused_op_3_kernel<bf16> grid=41x1x1 block=256x1x1 smem=0";
const LAUNCH_CONV: &str = "conv1d_update_kernel<1,0> grid=40x1x1 block=256x1x1 smem=0";
const LAUNCH_REC: &str =
    "cuda_recurrent_gated_delta_rule_kernel_128<0,4,0> grid=1x48x4 block=128x4x1 smem=0";
const LAUNCH_RMS: &str =
    "gated_rms_norm_kernel<32,half,bf16,float> grid=48x1x1 block=32x1x1 smem=0";

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/kernels/gdn")
}

fn load_meta() -> serde_json::Value {
    let path = data_dir().join("meta.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_slice(&bytes).unwrap()
}

fn read_bin(name: &str) -> Vec<u8> {
    let path = data_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut child = Command::new("sha256sum")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|err| panic!("spawn sha256sum: {err}"));
    {
        let stdin = child.stdin.as_mut().unwrap();
        use std::io::Write;
        stdin.write_all(bytes).unwrap();
    }
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "sha256sum: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_string()
}

/// Documented in `gdn_ref.py` and `meta.json` `lcg.algorithm`.
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

fn hex_prefix(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

fn diff_elements(got: &[u8], want: &[u8], width: usize) -> usize {
    assert_eq!(got.len(), want.len());
    assert_eq!(got.len() % width, 0);
    got.chunks(width)
        .zip(want.chunks(width))
        .filter(|(g, w)| g != w)
        .count()
}

fn assert_same(name: &str, got: &[u8], want: &[u8], width: usize) {
    let n = diff_elements(got, want, width);
    let elems = want.len() / width;
    assert_eq!(n, 0, "{name}: {n} of {elems} elements differ");
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

#[test]
fn lcg_matches_the_reference_state() {
    let meta_path = data_dir().join("meta.json");
    if !meta_path.is_file() {
        eprintln!("skipping lcg check: {} is missing", meta_path.display());
        return;
    }
    let meta = load_meta();
    let seed = meta["lcg"]["seed"].as_u64().unwrap() as u32;
    let count = meta["lcg"]["count"].as_u64().unwrap() as usize;
    assert_eq!(count, 48 * 128 * 128);
    let state = lcg_f32(seed, count);
    assert_eq!(
        hex_prefix(&state[..32]),
        meta["lcg"]["prefix_hex"].as_str().unwrap()
    );
    assert_eq!(sha256_hex(&state), meta["lcg"]["sha256"].as_str().unwrap());
    assert!(meta["total_bytes"].as_u64().unwrap() < 3_000_000);
}

#[test]
fn gdn_decode_kernels_match_the_fork() {
    if std::env::var("BERTH_GPU_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping gdn gpu test: set BERTH_GPU_TESTS=1");
        return;
    }
    if serve_blocks_gpu() {
        eprintln!("skipping gdn gpu test: port 8889 is listening or ss failed");
        return;
    }

    let meta = load_meta();
    let launches = &meta["launches"];
    assert_eq!(launches["ba"].as_str().unwrap(), LAUNCH_BA);
    assert_eq!(launches["fused"].as_str().unwrap(), LAUNCH_FUSED);
    assert_eq!(launches["conv"].as_str().unwrap(), LAUNCH_CONV);
    assert_eq!(launches["recurrent"].as_str().unwrap(), LAUNCH_REC);
    assert_eq!(launches["rms"].as_str().unwrap(), LAUNCH_RMS);
    let shapes = &meta["shapes"];
    assert_eq!(shapes["hidden"].as_i64().unwrap(), 5120);
    assert_eq!(shapes["f"].as_i64().unwrap(), 10240);
    assert_eq!(shapes["num_k_heads"].as_i64().unwrap(), 16);
    assert_eq!(shapes["num_v_heads"].as_i64().unwrap(), 48);
    assert_eq!(shapes["head_dim"].as_i64().unwrap(), 128);
    assert_eq!(shapes["conv_k"].as_i64().unwrap(), 4);
    assert_eq!(shapes["history_stride"].as_i64().unwrap(), 1);
    assert_eq!(shapes["ba_n"].as_i64().unwrap(), 96);

    let ctx = Context::new().unwrap();
    let stream = ctx.stream().unwrap();
    let st = stream.handle().as_ptr();

    let x = read_bin("x_ba.bin");
    let w_ba = read_bin("w_ba.bin");
    let bias = read_bin("bias_ba.bin");
    let y_want = read_bin("y_ba.bin");
    let y_bias_want = read_bin("y_ba_bias.bin");
    assert_eq!(x.len(), 5120 * 2);
    assert_eq!(w_ba.len(), 96 * 5120 * 2);
    assert_eq!(y_want.len(), 96 * 4);

    let x_d = DeviceBuffer::from_host(&stream, &x).unwrap();
    let w_ba_d = DeviceBuffer::from_host(&stream, &w_ba).unwrap();
    let bias_d = DeviceBuffer::from_host(&stream, &bias).unwrap();
    let y_d = DeviceBuffer::<u8>::zeroed(&stream, y_want.len()).unwrap();
    let y_bias_d = DeviceBuffer::<u8>::zeroed(&stream, y_bias_want.len()).unwrap();

    // SAFETY: each buffer is a live device allocation of the 27B shape, and
    // copy_to_host synchronizes `stream` before the buffer is read or dropped.
    unsafe {
        gdn::gdn_ba_gemv(
            x_d.ptr().as_u64() as *const c_void,
            w_ba_d.ptr().as_u64() as *const c_void,
            std::ptr::null(),
            y_d.ptr().as_u64() as *mut c_void,
            1,
            5120,
            96,
            st,
        )
    }
    .unwrap();
    assert_eq!(gdn::last_launch(), LAUNCH_BA);
    let mut y_got = vec![0u8; y_want.len()];
    y_d.copy_to_host(&stream, &mut y_got).unwrap();
    assert_same("y_ba", &y_got, &y_want, 4);

    unsafe {
        gdn::gdn_ba_gemv(
            x_d.ptr().as_u64() as *const c_void,
            w_ba_d.ptr().as_u64() as *const c_void,
            bias_d.ptr().as_u64() as *const c_void,
            y_bias_d.ptr().as_u64() as *mut c_void,
            1,
            5120,
            96,
            st,
        )
    }
    .unwrap();
    assert_eq!(gdn::last_launch(), LAUNCH_BA);
    let mut y_bias_got = vec![0u8; y_bias_want.len()];
    y_bias_d.copy_to_host(&stream, &mut y_bias_got).unwrap();
    assert_same("y_ba_bias", &y_bias_got, &y_bias_want, 4);

    let qkv = read_bin("qkv.bin");
    let dt_bias = read_bin("dt_bias.bin");
    let a_log = read_bin("a_log.bin");
    let mixed_want = read_bin("mixed_qkv.bin");
    let beta_want = read_bin("beta.bin");
    let g_want = read_bin("g.bin");
    assert_eq!(qkv.len(), 10240 * 4);
    assert_eq!(mixed_want.len(), 10240 * 2);
    assert_eq!(beta_want.len(), 48 * 2);
    assert_eq!(g_want.len(), 48 * 4);
    assert_eq!(dt_bias.len(), 48 * 2);
    assert_eq!(a_log.len(), 48 * 2);

    let qkv_d = DeviceBuffer::from_host(&stream, &qkv).unwrap();
    let dt_d = DeviceBuffer::from_host(&stream, &dt_bias).unwrap();
    let a_d = DeviceBuffer::from_host(&stream, &a_log).unwrap();
    let mixed_d = DeviceBuffer::<u8>::zeroed(&stream, mixed_want.len()).unwrap();
    let beta_d = DeviceBuffer::<u8>::zeroed(&stream, beta_want.len()).unwrap();
    let g_d = DeviceBuffer::<u8>::zeroed(&stream, g_want.len()).unwrap();
    // ba is the unbiased GEMV output, [1, 1, 96] in the same bytes as y.
    unsafe {
        gdn::gated_delta_net_fused_op_3(
            qkv_d.ptr().as_u64() as *const c_void,
            y_d.ptr().as_u64() as *const c_void,
            dt_d.ptr().as_u64() as *const c_void,
            a_d.ptr().as_u64() as *const c_void,
            mixed_d.ptr().as_u64() as *mut c_void,
            beta_d.ptr().as_u64() as *mut c_void,
            g_d.ptr().as_u64() as *mut c_void,
            1,
            1,
            10240,
            48,
            1.0,
            st,
        )
    }
    .unwrap();
    assert_eq!(gdn::last_launch(), LAUNCH_FUSED);
    let mut mixed_got = vec![0u8; mixed_want.len()];
    let mut beta_got = vec![0u8; beta_want.len()];
    let mut g_got = vec![0u8; g_want.len()];
    mixed_d.copy_to_host(&stream, &mut mixed_got).unwrap();
    beta_d.copy_to_host(&stream, &mut beta_got).unwrap();
    g_d.copy_to_host(&stream, &mut g_got).unwrap();
    assert_same("mixed_qkv", &mixed_got, &mixed_want, 2);
    assert_same("beta", &beta_got, &beta_want, 2);
    assert_same("g", &g_got, &g_want, 4);

    let w_conv = read_bin("w_conv.bin");
    let conv_out_zero = read_bin("conv_out_zero.bin");
    let conv_state_zero_out = read_bin("conv_state_zero_out.bin");
    let conv_state_nz = read_bin("conv_state_nz.bin");
    let conv_out_nz = read_bin("conv_out_nz.bin");
    let conv_state_nz_out = read_bin("conv_state_nz_out.bin");
    let slots = read_bin("slots_nz.bin");
    assert_eq!(w_conv.len(), 10240 * 4 * 2);
    assert_eq!(conv_state_nz.len(), 10240 * 4 * 2);
    assert_eq!(slots, 0i32.to_le_bytes());

    let w_conv_d = DeviceBuffer::from_host(&stream, &w_conv).unwrap();
    let conv_out_zero_d = DeviceBuffer::<u8>::zeroed(&stream, conv_out_zero.len()).unwrap();
    let state_zero_d = DeviceBuffer::<u8>::zeroed(&stream, conv_state_zero_out.len()).unwrap();
    unsafe {
        gdn::conv1d_update(
            mixed_d.ptr().as_u64() as *const c_void,
            state_zero_d.ptr().as_u64() as *mut c_void,
            std::ptr::null(),
            w_conv_d.ptr().as_u64() as *const c_void,
            std::ptr::null(),
            conv_out_zero_d.ptr().as_u64() as *mut c_void,
            1,
            10240,
            1,
            4,
            4,
            st,
        )
    }
    .unwrap();
    assert_eq!(gdn::last_launch(), LAUNCH_CONV);
    let mut conv_out_zero_got = vec![0u8; conv_out_zero.len()];
    let mut state_zero_got = vec![0u8; conv_state_zero_out.len()];
    conv_out_zero_d
        .copy_to_host(&stream, &mut conv_out_zero_got)
        .unwrap();
    state_zero_d
        .copy_to_host(&stream, &mut state_zero_got)
        .unwrap();
    assert_same("conv_out_zero", &conv_out_zero_got, &conv_out_zero, 2);
    assert_same(
        "conv_state_zero_out",
        &state_zero_got,
        &conv_state_zero_out,
        2,
    );

    let slots_d = DeviceBuffer::from_host(&stream, &slots).unwrap();
    let state_nz_d = DeviceBuffer::from_host(&stream, &conv_state_nz).unwrap();
    let conv_out_nz_d = DeviceBuffer::<u8>::zeroed(&stream, conv_out_nz.len()).unwrap();
    unsafe {
        gdn::conv1d_update(
            mixed_d.ptr().as_u64() as *const c_void,
            state_nz_d.ptr().as_u64() as *mut c_void,
            slots_d.ptr().as_u64() as *const c_void,
            w_conv_d.ptr().as_u64() as *const c_void,
            std::ptr::null(),
            conv_out_nz_d.ptr().as_u64() as *mut c_void,
            1,
            10240,
            1,
            4,
            4,
            st,
        )
    }
    .unwrap();
    assert_eq!(gdn::last_launch(), LAUNCH_CONV);
    let mut conv_out_nz_got = vec![0u8; conv_out_nz.len()];
    let mut state_nz_got = vec![0u8; conv_state_nz_out.len()];
    conv_out_nz_d
        .copy_to_host(&stream, &mut conv_out_nz_got)
        .unwrap();
    state_nz_d.copy_to_host(&stream, &mut state_nz_got).unwrap();
    assert_same("conv_out_nz", &conv_out_nz_got, &conv_out_nz, 2);
    assert_same("conv_state_nz_out", &state_nz_got, &conv_state_nz_out, 2);

    let core_zero = read_bin("core_zero.bin");
    let core_nz = read_bin("core_nz.bin");
    assert_eq!(core_zero.len(), 48 * 128 * 2);
    let state_bytes = 48 * 128 * 128 * 4;
    assert_eq!(state_bytes, 3_145_728);

    let rec_zero_d = DeviceBuffer::<u8>::zeroed(&stream, state_bytes).unwrap();
    let core_zero_d = DeviceBuffer::<u8>::zeroed(&stream, core_zero.len()).unwrap();
    unsafe {
        gdn::recurrent_gated_delta_rule(
            conv_out_zero_d.ptr().as_u64() as *const c_void,
            g_d.ptr().as_u64() as *const c_void,
            beta_d.ptr().as_u64() as *const c_void,
            rec_zero_d.ptr().as_u64() as *mut c_void,
            core_zero_d.ptr().as_u64() as *mut c_void,
            std::ptr::null(),
            1,
            1,
            16,
            48,
            128,
            128,
            1,
            st,
        )
    }
    .unwrap();
    assert_eq!(gdn::last_launch(), LAUNCH_REC);
    let mut core_zero_got = vec![0u8; core_zero.len()];
    let mut rec_zero_got = vec![0u8; state_bytes];
    core_zero_d
        .copy_to_host(&stream, &mut core_zero_got)
        .unwrap();
    rec_zero_d.copy_to_host(&stream, &mut rec_zero_got).unwrap();
    assert_same("core_zero", &core_zero_got, &core_zero, 2);
    let zero_hash = sha256_hex(&rec_zero_got);
    assert_eq!(
        zero_hash,
        meta["zero"]["recurrent_sha256"].as_str().unwrap(),
        "updated zero-state recurrent buffer differs (sha256 covers every f32 element)"
    );

    let seed = meta["lcg"]["seed"].as_u64().unwrap() as u32;
    let lcg = lcg_f32(seed, 48 * 128 * 128);
    let rec_nz_d = DeviceBuffer::from_host(&stream, &lcg).unwrap();
    let core_nz_d = DeviceBuffer::<u8>::zeroed(&stream, core_nz.len()).unwrap();
    unsafe {
        gdn::recurrent_gated_delta_rule(
            conv_out_nz_d.ptr().as_u64() as *const c_void,
            g_d.ptr().as_u64() as *const c_void,
            beta_d.ptr().as_u64() as *const c_void,
            rec_nz_d.ptr().as_u64() as *mut c_void,
            core_nz_d.ptr().as_u64() as *mut c_void,
            slots_d.ptr().as_u64() as *const c_void,
            1,
            1,
            16,
            48,
            128,
            128,
            1,
            st,
        )
    }
    .unwrap();
    assert_eq!(gdn::last_launch(), LAUNCH_REC);
    let mut core_nz_got = vec![0u8; core_nz.len()];
    let mut rec_nz_got = vec![0u8; state_bytes];
    core_nz_d.copy_to_host(&stream, &mut core_nz_got).unwrap();
    rec_nz_d.copy_to_host(&stream, &mut rec_nz_got).unwrap();
    assert_same("core_nz", &core_nz_got, &core_nz, 2);
    let nz_hash = sha256_hex(&rec_nz_got);
    assert_eq!(
        nz_hash,
        meta["nonzero"]["recurrent_sha256"].as_str().unwrap(),
        "updated nonzero recurrent buffer differs (sha256 covers every f32 element)"
    );

    let w_norm = read_bin("w_norm.bin");
    let z = read_bin("z.bin");
    let y_rms_zero = read_bin("y_rms_zero.bin");
    let y_rms_nz = read_bin("y_rms_nz.bin");
    assert_eq!(w_norm.len(), 128 * 2);
    assert_eq!(z.len(), 48 * 128 * 4);
    assert_eq!(y_rms_zero.len(), 48 * 128 * 2);

    let w_norm_d = DeviceBuffer::from_host(&stream, &w_norm).unwrap();
    let z_d = DeviceBuffer::from_host(&stream, &z).unwrap();
    let y0_d = DeviceBuffer::<u8>::zeroed(&stream, y_rms_zero.len()).unwrap();
    let y1_d = DeviceBuffer::<u8>::zeroed(&stream, y_rms_nz.len()).unwrap();
    unsafe {
        gdn::gated_rms_norm(
            core_zero_d.ptr().as_u64() as *const c_void,
            w_norm_d.ptr().as_u64() as *const c_void,
            y0_d.ptr().as_u64() as *mut c_void,
            z_d.ptr().as_u64() as *const c_void,
            1e-6,
            0.0,
            48,
            128,
            1,
            0,
            0,
            st,
        )
    }
    .unwrap();
    assert_eq!(gdn::last_launch(), LAUNCH_RMS);
    unsafe {
        gdn::gated_rms_norm(
            core_nz_d.ptr().as_u64() as *const c_void,
            w_norm_d.ptr().as_u64() as *const c_void,
            y1_d.ptr().as_u64() as *mut c_void,
            z_d.ptr().as_u64() as *const c_void,
            1e-6,
            0.0,
            48,
            128,
            1,
            0,
            0,
            st,
        )
    }
    .unwrap();
    assert_eq!(gdn::last_launch(), LAUNCH_RMS);
    let mut y0_got = vec![0u8; y_rms_zero.len()];
    let mut y1_got = vec![0u8; y_rms_nz.len()];
    y0_d.copy_to_host(&stream, &mut y0_got).unwrap();
    y1_d.copy_to_host(&stream, &mut y1_got).unwrap();
    assert_same("y_rms_zero", &y0_got, &y_rms_zero, 2);
    assert_same("y_rms_nz", &y1_got, &y_rms_nz, 2);
}
