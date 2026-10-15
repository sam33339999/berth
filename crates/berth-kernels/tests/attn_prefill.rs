//! Bit-exact GPU checks for the full-attention prefill core.
//!
//! `tests/attn_prefill_ref.py` writes under `CARGO_TARGET_TMPDIR`. Each case
//! is the fork op on seeded fp16 input, with the layer-3 q/k norm weights.
//! The cases hit every captured specialization class (qc and fp16 staging,
//! q_len other and divisible by 16, page counts 1, 6, and 16, plus the
//! longer chunks). Outputs and the written cq3 pages are compared, 0
//! differing.
//!
//! Runs only with `BERTH_GPU_TESTS=1`. Skips when `:8889` is listening.

use std::path::{Path, PathBuf};
use std::process::Command;

use berth_cuda::{Context, DeviceBuffer, Stream};
use berth_kernels::attn::TRITON_CUBIN_BUILD as DECODE_BUILD;
use berth_kernels::attn_prefill::{
    dequant_cache_paged_window, needs_staging, prefill_cubin_key, prefill_num_splits,
    staging_pages, DequantWindow, PagedPrefill, PrefillChunk, SplitShape, HEAD_DIM, KV_HEADS,
    PAGE_SIZE, Q_HEADS, TOKEN_DIM,
};

const REF_PY: &str = "/home/sam/qwen38-exl3-dflash2/env/bin/python";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn cubin_dir() -> PathBuf {
    workspace().join("target/triton-cubins")
}

fn port_8889_listening() -> bool {
    let Ok(out) = Command::new("ss").args(["-ltn"]).output() else {
        return true;
    };
    String::from_utf8_lossy(&out.stdout).contains(":8889 ")
        || String::from_utf8_lossy(&out.stderr).contains(":8889 ")
}

fn read_bin(dir: &Path, name: &str) -> Vec<u8> {
    let path = dir.join(name);
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

fn download(stream: &Stream, buf: &DeviceBuffer<u8>) -> Vec<u8> {
    let mut got = vec![0u8; buf.len()];
    buf.copy_to_host(stream, &mut got).unwrap();
    got
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
        .filter(|(a, b)| a != b)
        .count()
}

fn check(failures: &mut Vec<String>, name: &str, what: &str, got: &[u8], want: &[u8], elem: usize) {
    let n = want.len() / elem;
    let differing = differ(got, want, elem);
    eprintln!("{name}: {what}, {differing} of {n} elements differ");
    if differing != 0 {
        failures.push(format!(
            "{name}: {what}, {differing} of {n} elements differ"
        ));
    }
}

fn nonzero(bytes: &[u8], what: &str) {
    assert!(bytes.iter().any(|b| *b != 0), "{what} is all zeros");
}

fn fp16_bytes(q_len: usize, heads: usize) -> usize {
    q_len * heads * HEAD_DIM as usize * 2
}

fn page_bytes(pages: usize) -> usize {
    let words = (TOKEN_DIM / 32 * 3) as usize;
    pages * PAGE_SIZE as usize * words * 4
}

fn scale_bytes(pages: usize) -> usize {
    let groups = (TOKEN_DIM / 32) as usize;
    pages * PAGE_SIZE as usize * groups * 2
}

fn dev(buf: &DeviceBuffer<u8>) -> u64 {
    buf.ptr().as_u64()
}

#[derive(Clone, Copy)]
struct Case<'a> {
    name: &'a str,
    q_len: i32,
    past_len: i32,
    pages: i32,
    cache_in: bool,
}

fn run_case(
    prefill: &PagedPrefill,
    stream: &Stream,
    dir: &Path,
    case: &Case<'_>,
    shared: &Shared,
    failures: &mut Vec<String>,
) {
    let Case {
        name,
        q_len,
        past_len,
        pages,
        cache_in,
    } = *case;
    let q_len_u = q_len as usize;
    let pages_u = pages as usize;
    let qg = upload(stream, &read_bin(dir, &format!("{name}_qg.bin")));
    let k = upload(stream, &read_bin(dir, &format!("{name}_k.bin")));
    let v = upload(stream, &read_bin(dir, &format!("{name}_v.bin")));
    assert_eq!(qg.len(), fp16_bytes(q_len_u, Q_HEADS as usize * 2));
    assert_eq!(k.len(), fp16_bytes(q_len_u, KV_HEADS as usize));
    assert_eq!(v.len(), k.len());

    let q = zeros(stream, fp16_bytes(q_len_u, Q_HEADS as usize));
    let g = zeros(stream, q.len());
    let o = zeros(stream, q.len());

    let (k_pages, v_pages, k_scales, v_scales) = if cache_in {
        let k_pages = upload(stream, &read_bin(dir, &format!("{name}_k_pages_in.bin")));
        let v_pages = upload(stream, &read_bin(dir, &format!("{name}_v_pages_in.bin")));
        let k_scales = upload(stream, &read_bin(dir, &format!("{name}_k_scales_in.bin")));
        let v_scales = upload(stream, &read_bin(dir, &format!("{name}_v_scales_in.bin")));
        (k_pages, v_pages, k_scales, v_scales)
    } else {
        (
            zeros(stream, page_bytes(pages_u)),
            zeros(stream, page_bytes(pages_u)),
            zeros(stream, scale_bytes(pages_u)),
            zeros(stream, scale_bytes(pages_u)),
        )
    };
    assert_eq!(k_pages.len(), page_bytes(pages_u));
    assert_eq!(v_pages.len(), page_bytes(pages_u));
    assert_eq!(k_scales.len(), scale_bytes(pages_u));
    assert_eq!(v_scales.len(), scale_bytes(pages_u));

    let table = upload(stream, &i32_bytes(&(0..pages).collect::<Vec<_>>()));
    let seqlens = upload(stream, &i32_bytes(&[past_len]));
    let (scratch_k, scratch_v, identity) = if needs_staging(q_len) {
        let elems = berth_kernels::attn_prefill::staging_scratch_elems(1, pages).unwrap();
        let ident = upload(stream, &i32_bytes(&(0..pages).collect::<Vec<_>>()));
        (
            zeros(stream, elems * 2),
            zeros(stream, elems * 2),
            Some(ident),
        )
    } else {
        (zeros(stream, 0), zeros(stream, 0), None)
    };
    let (global_n, profile_n) = prefill.scratch_bytes(q_len);
    let global = zeros(stream, global_n as usize);
    let profile = zeros(stream, profile_n as usize);

    let args = PrefillChunk {
        qg: dev(&qg),
        k: dev(&k),
        v: dev(&v),
        q: dev(&q),
        g: dev(&g),
        o: dev(&o),
        k_cache: dev(&k_pages),
        v_cache: dev(&v_pages),
        k_scales: dev(&k_scales),
        v_scales: dev(&v_scales),
        block_table: dev(&table),
        cache_seqlens: dev(&seqlens),
        inv_freq: dev(&shared.inv_freq),
        q_norm: dev(&shared.q_norm),
        k_norm: dev(&shared.k_norm),
        h32: dev(&shared.h32),
        scratch_k: dev(&scratch_k),
        scratch_v: dev(&scratch_v),
        identity_table: identity.as_ref().map(dev).unwrap_or(0),
        global_scratch: dev(&global),
        profile_scratch: dev(&profile),
    };
    unsafe {
        prefill
            .prefill_chunk(stream, &args, q_len, pages, past_len)
            .unwrap()
    };

    check(
        failures,
        name,
        "output",
        &download(stream, &o),
        &read_bin(dir, &format!("{name}_o.bin")),
        2,
    );
    check(
        failures,
        name,
        "k pages",
        &download(stream, &k_pages),
        &read_bin(dir, &format!("{name}_k_pages.bin")),
        4,
    );
    check(
        failures,
        name,
        "v pages",
        &download(stream, &v_pages),
        &read_bin(dir, &format!("{name}_v_pages.bin")),
        4,
    );
    check(
        failures,
        name,
        "k scales",
        &download(stream, &k_scales),
        &read_bin(dir, &format!("{name}_k_scales.bin")),
        2,
    );
    check(
        failures,
        name,
        "v scales",
        &download(stream, &v_scales),
        &read_bin(dir, &format!("{name}_v_scales.bin")),
        2,
    );
}

struct Shared {
    inv_freq: DeviceBuffer<u8>,
    q_norm: DeviceBuffer<u8>,
    k_norm: DeviceBuffer<u8>,
    h32: DeviceBuffer<u8>,
}

#[test]
fn prefill_matches_fork() {
    if std::env::var("BERTH_GPU_TESTS").ok().as_deref() != Some("1") {
        return;
    }
    if port_8889_listening() {
        eprintln!("attn prefill: skipping GPU test; port 8889 is in use or ss failed");
        return;
    }
    let root = workspace();
    let out = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("attn-prefill");
    std::fs::create_dir_all(&out).unwrap();
    let status = Command::new(REF_PY)
        .arg(root.join("crates/berth-kernels/tests/attn_prefill_ref.py"))
        .arg(&out)
        .current_dir(&root)
        .env("PYTHONPATH", "/home/sam/qwen38-exl3-dflash2/exllamav3")
        .env("TRITON_LIBCUDA_PATH", "/usr/lib/x86_64-linux-gnu")
        .env("EXL3_QC_STAGING", "1")
        .status()
        .unwrap_or_else(|err| panic!("failed to run reference: {err}"));
    assert!(status.success(), "reference failed: {status}");

    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(manifest["seed"].as_i64().unwrap(), 20261015);

    let ctx = Context::new().unwrap();
    let stream = ctx.stream().unwrap();
    let prefill = PagedPrefill::load(&ctx, &cubin_dir()).unwrap_or_else(|err| {
        panic!("{err}\nRun: {DECODE_BUILD}");
    });
    let shared = Shared {
        inv_freq: upload(&stream, &read_bin(&out, "inv_freq.bin")),
        q_norm: upload(&stream, &read_bin(&out, "q_norm.bin")),
        k_norm: upload(&stream, &read_bin(&out, "k_norm.bin")),
        h32: upload(&stream, &read_bin(&out, "h32.bin")),
    };
    assert_eq!(shared.inv_freq.len(), 32 * 4);
    assert_eq!(shared.q_norm.len(), HEAD_DIM as usize * 2);
    assert_eq!(shared.k_norm.len(), HEAD_DIM as usize * 2);
    assert_eq!(shared.h32.len(), 32 * 32 * 2);
    nonzero(&read_bin(&out, "q_norm.bin"), "q_norm");
    nonzero(&read_bin(&out, "k_norm.bin"), "k_norm");

    let cubin_manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(cubin_dir().join("manifest.json")).unwrap())
            .unwrap();
    let mut captured = std::collections::BTreeSet::new();
    for key in cubin_manifest["prefill"]["classes"].as_array().unwrap() {
        captured.insert(key.as_str().unwrap().to_string());
    }
    let mut hit = std::collections::BTreeSet::new();

    let mut failures = Vec::new();
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let q_len = i32::try_from(case["q_len"].as_i64().unwrap()).unwrap();
        let past_len = i32::try_from(case["past_len"].as_i64().unwrap()).unwrap();
        let pages = i32::try_from(case["pages"].as_i64().unwrap()).unwrap();
        let cache_in = case
            .get("cache_in")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let end = i64::from(past_len) + i64::from(q_len);
        assert!(end <= i64::from(pages) * i64::from(PAGE_SIZE));
        assert!(staging_pages(1, pages).is_ok());
        let splits = prefill_num_splits(&SplitShape {
            q_len,
            bt_width: pages,
            page_size: PAGE_SIZE,
            kv_append_len: q_len,
            block_m: 64,
            n_q_heads: Q_HEADS,
            bsz: 1,
            hd_pad: HEAD_DIM,
            sm_count: 82,
        })
        .unwrap();
        assert_eq!(splits, 1, "{name}");
        hit.insert(prefill_cubin_key(
            needs_staging(q_len),
            1,
            q_len,
            q_len,
            pages,
            -1,
            -1,
        ));
        run_case(
            &prefill,
            &stream,
            &out,
            &Case {
                name,
                q_len,
                past_len,
                pages,
                cache_in,
            },
            &shared,
            &mut failures,
        );
    }
    assert_eq!(
        hit, captured,
        "every captured prefill class must be launched"
    );
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn dequant_null_is_rejected() {
    let rc = unsafe {
        dequant_cache_paged_window(
            &DequantWindow {
                k_in: std::ptr::null(),
                k_scales: std::ptr::null(),
                k_out: std::ptr::null_mut(),
                v_in: std::ptr::null(),
                v_scales: std::ptr::null(),
                v_out: std::ptr::null_mut(),
                cache_seqlens: std::ptr::null(),
                block_table: std::ptr::null(),
                bsz: 1,
                pages_per_seq: 1,
                token_dim: TOKEN_DIM,
                k_bits: 3,
                v_bits: 3,
                page_size: PAGE_SIZE,
                bonus_len: 1,
                compand_a: 0.0,
            },
            std::ptr::null_mut(),
        )
    };
    let err = rc.expect_err("null");
    assert!(err.to_string().contains("null"), "{err}");
}
