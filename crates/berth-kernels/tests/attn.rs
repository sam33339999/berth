//! Bit-exact GPU checks for the full-attention decode kernels.
//!
//! `tests/attn_ref.py` writes `testdata/kernels/attn/`. Each case is the fork
//! op on seeded fp16/bf16 input: `deinterleave_qg`, `rope` (NEOX, bf16 norms),
//! `quant_cache_paged` (cq3), `mul_sigmoid_`, and the whole core
//! (deinterleave, rope, cache write, paged split, paged combine, gate) at
//! position 0 and at position 300. The paged pair is the cubin in
//! `target/triton-cubins/`; `TritonKernel::launch` is not a Python binding.
//!
//! Runs only with `BERTH_GPU_TESTS=1`. Skips when `:8889` is listening.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;

use berth_cuda::{Context, DeviceBuffer, Stream};
use berth_kernels::attn::{
    deinterleave_qg, mul_sigmoid, quant_cache_paged, rope, split_config, splits_cap, PagedDecode,
    PagedDecodeArgs, QuantCacheLaunch, RopeLaunch,
};

const REF_PY: &str = "\
PYTHONPATH=/home/sam/qwen38-exl3-dflash2/exllamav3 \
TRITON_LIBCUDA_PATH=/usr/lib/x86_64-linux-gnu \
/home/sam/qwen38-exl3-dflash2/env/bin/python \
crates/berth-kernels/tests/attn_ref.py";

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/kernels/attn")
}

fn cubin_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins")
}

fn port_8889_listening() -> bool {
    let Ok(out) = Command::new("ss").args(["-ltn"]).output() else {
        return true;
    };
    String::from_utf8_lossy(&out.stdout).contains(":8889 ")
        || String::from_utf8_lossy(&out.stderr).contains(":8889 ")
}

fn read_bin(name: &str) -> Vec<u8> {
    let path = data_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

fn upload(stream: &Stream, bytes: &[u8]) -> DeviceBuffer<u8> {
    DeviceBuffer::from_host(stream, bytes).unwrap()
}

fn zeros(stream: &Stream, len: usize) -> DeviceBuffer<u8> {
    DeviceBuffer::<u8>::zeroed(stream, len).unwrap()
}

fn i32_bytes(vals: &[i32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn json_i32s(v: &serde_json::Value) -> Vec<i32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|item| i32::try_from(item.as_i64().unwrap()).unwrap())
        .collect()
}

fn file_name(v: &serde_json::Value, key: &str) -> String {
    v[key]
        .as_str()
        .unwrap_or_else(|| panic!("{key} is not a file name"))
        .to_string()
}

fn differ(got: &[u8], want: &[u8], elem: usize) -> usize {
    assert_eq!(
        got.len(),
        want.len(),
        "byte length {} != {}",
        got.len(),
        want.len()
    );
    assert_eq!(got.len() % elem, 0);
    got.chunks(elem)
        .zip(want.chunks(elem))
        .filter(|(g, w)| g != w)
        .count()
}

fn check(
    failures: &mut Vec<String>,
    name: &str,
    kernel: &str,
    got: &[u8],
    want: &[u8],
    elem: usize,
) {
    let n = want.len() / elem;
    let differing = differ(got, want, elem);
    eprintln!("{name}: {kernel}, {differing} of {n} elements differ");
    if differing != 0 {
        failures.push(format!(
            "{name}: {kernel}, {differing} of {n} elements differ"
        ));
    }
}

fn download(stream: &Stream, buf: &DeviceBuffer<u8>) -> Vec<u8> {
    let mut got = vec![0u8; buf.len()];
    buf.copy_to_host(stream, &mut got).unwrap();
    got
}

fn rope_27b(
    q: &DeviceBuffer<u8>,
    k: &DeviceBuffer<u8>,
    inv_freq: &DeviceBuffer<u8>,
    q_norm: &DeviceBuffer<u8>,
    k_norm: &DeviceBuffer<u8>,
    position: u32,
) -> RopeLaunch {
    RopeLaunch {
        q: q.ptr().as_u64() as *const c_void,
        out_q: q.ptr().as_u64() as *mut c_void,
        k: k.ptr().as_u64() as *const c_void,
        out_k: k.ptr().as_u64() as *mut c_void,
        inv_freq: inv_freq.ptr().as_u64() as *const c_void,
        bsz: 1,
        seq_len: 1,
        num_heads_q: 24,
        num_heads_k: 4,
        head_dim: 256,
        q_head_stride: 256,
        k_head_stride: 256,
        partial_head_dim: 64,
        position,
        positions: std::ptr::null(),
        position_ids: std::ptr::null(),
        rope_mode: 2,
        attn_factor: 1.0,
        q_norm: q_norm.ptr().as_u64() as *const c_void,
        k_norm: k_norm.ptr().as_u64() as *const c_void,
        norm_eps: 1e-6,
        norm_constant_bias: 1.0,
        inv_freq_table: false,
        inv_freq_stride: 0,
        llama_4_scaling_beta: 0.0,
        llama_4_scaling_original: 0,
        position_ids_stride: 1,
        rotate_dims: 1,
        rotate_offset: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn quant_27b(
    k_in: &DeviceBuffer<u8>,
    k_out: &DeviceBuffer<u8>,
    k_scales: &DeviceBuffer<u8>,
    v_in: &DeviceBuffer<u8>,
    v_out: &DeviceBuffer<u8>,
    v_scales: &DeviceBuffer<u8>,
    seqlens: &DeviceBuffer<u8>,
    table: &DeviceBuffer<u8>,
    seq_len: i32,
    blocks_per_seq: i32,
) -> QuantCacheLaunch {
    QuantCacheLaunch {
        k_in: k_in.ptr().as_u64() as *const c_void,
        k_out: k_out.ptr().as_u64() as *mut c_void,
        k_scales: k_scales.ptr().as_u64() as *mut c_void,
        v_in: v_in.ptr().as_u64() as *const c_void,
        v_out: v_out.ptr().as_u64() as *mut c_void,
        v_scales: v_scales.ptr().as_u64() as *mut c_void,
        cache_seqlens: seqlens.ptr().as_u64() as *const c_void,
        block_table: table.ptr().as_u64() as *const c_void,
        bsz: 1,
        seq_len,
        blocks_per_seq,
        token_dim: 1024,
        k_bits: 3,
        v_bits: 3,
        page_size: 256,
        compand_a: 0.0,
        in_contiguous: true,
    }
}

struct CacheOut {
    o: Vec<u8>,
    k_pages: Vec<u8>,
    v_pages: Vec<u8>,
    k_scales: Vec<u8>,
    v_scales: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
fn run_core(
    stream: &Stream,
    raw: *mut c_void,
    decode: &PagedDecode,
    splits: i32,
    inv_freq: &DeviceBuffer<u8>,
    q_norm: &DeviceBuffer<u8>,
    k_norm: &DeviceBuffer<u8>,
    h32: &DeviceBuffer<u8>,
    case: &serde_json::Value,
    k_pages_in: &[u8],
    v_pages_in: &[u8],
    k_scales_in: &[u8],
    v_scales_in: &[u8],
) -> CacheOut {
    let qg_b = read_bin(&file_name(case, "qg"));
    let k_b = read_bin(&file_name(case, "k"));
    let v_b = read_bin(&file_name(case, "v"));
    let qg = upload(stream, &qg_b);
    let q = zeros(stream, 24 * 256 * 2);
    let g = zeros(stream, 24 * 256 * 2);
    let k = upload(stream, &k_b);
    let v = upload(stream, &v_b);
    let k_pages = upload(stream, k_pages_in);
    let v_pages = upload(stream, v_pages_in);
    let k_scales = upload(stream, k_scales_in);
    let v_scales = upload(stream, v_scales_in);
    let table = upload(stream, &i32_bytes(&json_i32s(&case["block_table"])));
    let seqlens = upload(stream, &i32_bytes(&json_i32s(&case["cache_seqlens"])));
    let o = zeros(stream, 24 * 256 * 2);
    let po = zeros(stream, decode.partial_o_elems(1, splits).unwrap() * 4);
    let ml = zeros(stream, decode.partial_ml_elems(1, splits).unwrap() * 4);

    let position = u32::try_from(case["position"].as_u64().unwrap()).unwrap();
    let bt_width = i32::try_from(case["bt_width"].as_u64().unwrap()).unwrap();

    // SAFETY: every buffer is a device allocation of the 27B decode shape and
    // stays alive until `stream` is synchronized by `download`.
    unsafe {
        deinterleave_qg(
            qg.ptr().as_u64() as *const c_void,
            q.ptr().as_u64() as *mut c_void,
            g.ptr().as_u64() as *mut c_void,
            256,
            24 * 256,
            raw,
        )
        .unwrap();
        rope(&rope_27b(&q, &k, inv_freq, q_norm, k_norm, position), raw).unwrap();
        quant_cache_paged(
            &quant_27b(
                &k, &k_pages, &k_scales, &v, &v_pages, &v_scales, &seqlens, &table, 1, bt_width,
            ),
            raw,
        )
        .unwrap();
        decode
            .launch(
                stream,
                &PagedDecodeArgs {
                    q: q.ptr().as_u64(),
                    k_cache: k_pages.ptr().as_u64(),
                    v_cache: v_pages.ptr().as_u64(),
                    block_table: table.ptr().as_u64(),
                    cache_seqlens: seqlens.ptr().as_u64(),
                    out: o.ptr().as_u64(),
                    partial_o: po.ptr().as_u64(),
                    partial_ml: ml.ptr().as_u64(),
                    k_scales: k_scales.ptr().as_u64(),
                    v_scales: v_scales.ptr().as_u64(),
                    h32: h32.ptr().as_u64(),
                    sinks: q.ptr().as_u64(),
                },
                1,
                bt_width,
                splits,
            )
            .unwrap();
        mul_sigmoid(
            o.ptr().as_u64() as *mut c_void,
            g.ptr().as_u64() as *const c_void,
            24 * 256,
            raw,
        )
        .unwrap();
    }

    CacheOut {
        o: download(stream, &o),
        k_pages: download(stream, &k_pages),
        v_pages: download(stream, &v_pages),
        k_scales: download(stream, &k_scales),
        v_scales: download(stream, &v_scales),
    }
}

#[test]
fn attn_matches_the_fork_bit_for_bit() {
    if std::env::var("BERTH_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping attn: set BERTH_GPU_TESTS=1");
        return;
    }
    if port_8889_listening() {
        eprintln!("skipping attn: :8889 is listening");
        return;
    }
    let meta_path = data_dir().join("meta.json");
    if !meta_path.is_file() {
        panic!("testdata missing at {}. Run: {REF_PY}", meta_path.display());
    }
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&meta_path).unwrap()).unwrap();

    let ctx = Context::new().unwrap();
    let stream = ctx.stream().unwrap();
    let raw = stream.handle().as_ptr();

    let mut failures = Vec::new();

    let qg = upload(&stream, &read_bin(&file_name(&meta["deinterleave"], "qg")));
    let q = zeros(&stream, 24 * 256 * 2);
    let g = zeros(&stream, 24 * 256 * 2);
    unsafe {
        deinterleave_qg(
            qg.ptr().as_u64() as *const c_void,
            q.ptr().as_u64() as *mut c_void,
            g.ptr().as_u64() as *mut c_void,
            256,
            24 * 256,
            raw,
        )
        .unwrap();
    }
    check(
        &mut failures,
        "deinterleave q",
        "deinterleave_qg_kernel",
        &download(&stream, &q),
        &read_bin(&file_name(&meta["deinterleave"], "q")),
        2,
    );
    check(
        &mut failures,
        "deinterleave g",
        "deinterleave_qg_kernel",
        &download(&stream, &g),
        &read_bin(&file_name(&meta["deinterleave"], "g")),
        2,
    );

    let inv_freq = upload(&stream, &read_bin(&file_name(&meta["shared"], "inv_freq")));
    let q_norm = upload(&stream, &read_bin(&file_name(&meta["shared"], "q_norm")));
    let k_norm = upload(&stream, &read_bin(&file_name(&meta["shared"], "k_norm")));
    let h32 = upload(&stream, &read_bin(&file_name(&meta["shared"], "h32")));
    let rq = upload(&stream, &read_bin(&file_name(&meta["rope"], "q_in")));
    let rk = upload(&stream, &read_bin(&file_name(&meta["rope"], "k_in")));
    unsafe {
        rope(&rope_27b(&rq, &rk, &inv_freq, &q_norm, &k_norm, 300), raw).unwrap();
    }
    check(
        &mut failures,
        "rope q",
        "rope_kernel<2,1>",
        &download(&stream, &rq),
        &read_bin(&file_name(&meta["rope"], "q_out")),
        2,
    );
    check(
        &mut failures,
        "rope k",
        "rope_kernel<2,1>",
        &download(&stream, &rk),
        &read_bin(&file_name(&meta["rope"], "k_out")),
        2,
    );

    let x = upload(&stream, &read_bin(&file_name(&meta["mul_sigmoid"], "x_in")));
    let y = upload(&stream, &read_bin(&file_name(&meta["mul_sigmoid"], "y")));
    unsafe {
        mul_sigmoid(
            x.ptr().as_u64() as *mut c_void,
            y.ptr().as_u64() as *const c_void,
            24 * 256,
            raw,
        )
        .unwrap();
    }
    check(
        &mut failures,
        "mul_sigmoid",
        "mul_sigmoid_kernel_h",
        &download(&stream, &x),
        &read_bin(&file_name(&meta["mul_sigmoid"], "x_out")),
        2,
    );

    let quant = &meta["quant"];
    let k_in = upload(&stream, &read_bin(&file_name(quant, "k_in")));
    let v_in = upload(&stream, &read_bin(&file_name(quant, "v_in")));
    let k_pages_b = read_bin(&file_name(quant, "k_pages"));
    let v_pages_b = read_bin(&file_name(quant, "v_pages"));
    let k_scales_b = read_bin(&file_name(quant, "k_scales"));
    let v_scales_b = read_bin(&file_name(quant, "v_scales"));
    let k_pages = zeros(&stream, k_pages_b.len());
    let v_pages = zeros(&stream, v_pages_b.len());
    let k_scales = zeros(&stream, k_scales_b.len());
    let v_scales = zeros(&stream, v_scales_b.len());
    let table = upload(&stream, &i32_bytes(&json_i32s(&quant["block_table"])));
    let seqlens = upload(&stream, &i32_bytes(&json_i32s(&quant["cache_seqlens"])));
    unsafe {
        quant_cache_paged(
            &quant_27b(
                &k_in, &k_pages, &k_scales, &v_in, &v_pages, &v_scales, &seqlens, &table, 300, 2,
            ),
            raw,
        )
        .unwrap();
    }
    check(
        &mut failures,
        "quant k pages",
        "quant_cache_paged_kernel<3,3>",
        &download(&stream, &k_pages),
        &k_pages_b,
        4,
    );
    check(
        &mut failures,
        "quant v pages",
        "quant_cache_paged_kernel<3,3>",
        &download(&stream, &v_pages),
        &v_pages_b,
        4,
    );
    check(
        &mut failures,
        "quant k scales",
        "quant_cache_paged_kernel<3,3>",
        &download(&stream, &k_scales),
        &k_scales_b,
        2,
    );
    check(
        &mut failures,
        "quant v scales",
        "quant_cache_paged_kernel<3,3>",
        &download(&stream, &v_scales),
        &v_scales_b,
        2,
    );

    let decode = PagedDecode::load(&ctx, &cubin_dir()).unwrap_or_else(|err| panic!("{err}"));
    let sm = i32::try_from(meta["sm_count"].as_u64().unwrap()).unwrap();
    let programs = decode.programs(1).unwrap();
    let cap = splits_cap(programs, sm).unwrap();
    assert_eq!(programs, 4, "programs");
    assert_eq!(
        cap,
        i32::try_from(meta["splits_cap"].as_u64().unwrap()).unwrap()
    );
    let (empty_ns, empty_sl) = split_config(1, 256, 1, 32, cap).unwrap();
    let (cached_ns, cached_sl) = split_config(2, 256, 1, 32, cap).unwrap();
    let meta_i32 = |case: &str, key: &str| -> i32 {
        i32::try_from(meta[case][key].as_i64().unwrap()).unwrap()
    };
    assert_eq!(
        (empty_ns, empty_sl),
        (
            meta_i32("core_empty", "num_splits"),
            meta_i32("core_empty", "split_len")
        )
    );
    assert_eq!(
        (cached_ns, cached_sl),
        (
            meta_i32("core_cached", "num_splits"),
            meta_i32("core_cached", "split_len")
        )
    );

    let chain = "deinterleave_qg + rope_kernel<2,1> + quant_cache_paged_kernel<3,3> + _paged_attn_decode_split_kernel + _paged_attn_decode_combine_kernel + mul_sigmoid_kernel_h";

    let empty_pages = 256 * 96 * 4;
    let empty_scales = 256 * 32 * 2;
    let empty = run_core(
        &stream,
        raw,
        &decode,
        cap,
        &inv_freq,
        &q_norm,
        &k_norm,
        &h32,
        &meta["core_empty"],
        &vec![0u8; empty_pages],
        &vec![0u8; empty_pages],
        &vec![0u8; empty_scales],
        &vec![0u8; empty_scales],
    );
    let case = &meta["core_empty"];
    check(
        &mut failures,
        "core_empty o",
        chain,
        &empty.o,
        &read_bin(&file_name(case, "o")),
        2,
    );
    check(
        &mut failures,
        "core_empty k pages",
        chain,
        &empty.k_pages,
        &read_bin(&file_name(case, "k_pages")),
        4,
    );
    check(
        &mut failures,
        "core_empty v pages",
        chain,
        &empty.v_pages,
        &read_bin(&file_name(case, "v_pages")),
        4,
    );
    check(
        &mut failures,
        "core_empty k scales",
        chain,
        &empty.k_scales,
        &read_bin(&file_name(case, "k_scales")),
        2,
    );
    check(
        &mut failures,
        "core_empty v scales",
        chain,
        &empty.v_scales,
        &read_bin(&file_name(case, "v_scales")),
        2,
    );

    let case = &meta["core_cached"];
    let cached = run_core(
        &stream,
        raw,
        &decode,
        cap,
        &inv_freq,
        &q_norm,
        &k_norm,
        &h32,
        case,
        &k_pages_b,
        &v_pages_b,
        &k_scales_b,
        &v_scales_b,
    );
    check(
        &mut failures,
        "core_cached o",
        chain,
        &cached.o,
        &read_bin(&file_name(case, "o")),
        2,
    );
    check(
        &mut failures,
        "core_cached k pages",
        chain,
        &cached.k_pages,
        &read_bin(&file_name(case, "k_pages")),
        4,
    );
    check(
        &mut failures,
        "core_cached v pages",
        chain,
        &cached.v_pages,
        &read_bin(&file_name(case, "v_pages")),
        4,
    );
    check(
        &mut failures,
        "core_cached k scales",
        chain,
        &cached.k_scales,
        &read_bin(&file_name(case, "k_scales")),
        2,
    );
    check(
        &mut failures,
        "core_cached v scales",
        chain,
        &cached.v_scales,
        &read_bin(&file_name(case, "v_scales")),
        2,
    );

    assert!(failures.is_empty(), "{failures:#?}");
}
