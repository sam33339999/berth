//! Sampler kernels against the fork reference in `testdata/kernels/sample`.
//!
//! `tests/sample_ref.py` writes `meta.json` and the DRY breaker list. Logit
//! rows and histories are regenerated from the LCG in that file; the test
//! compares blake2b-256 of the kernel output, or the fused token id.
//!
//! The GPU test runs only with `BERTH_GPU_TESTS=1` and skips when port 8889
//! is listening. It does not load the 27B. A null-pointer check and the
//! blake2b self-check run without a device.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::process::Command;

use berth_cuda::{Context, DeviceBuffer, Stream};
use berth_kernels::sample;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/kernels/sample")
}

fn load_meta() -> serde_json::Value {
    let path = data_dir().join("meta.json");
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()))
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

fn lcg_next(x: u32) -> u32 {
    x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
}

fn lcg_row(seed: u32, n: usize, dtype: &str) -> Vec<u8> {
    let mut x = seed;
    match dtype {
        "f16" => {
            let mut out = vec![0u8; n * 2];
            for i in 0..n {
                x = lcg_next(x);
                let mant = (x >> 8) & 0x3ff;
                let exp = 14 + ((x >> 20) & 3);
                let sign = x >> 31;
                let bits = ((sign << 15) | (exp << 10) | mant) as u16;
                out[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            }
            out
        }
        "f32" => {
            let mut out = vec![0u8; n * 4];
            for i in 0..n {
                x = lcg_next(x);
                let mut bits = (x & 0x007f_ffff) | 0x3f80_0000;
                if x & 1 != 0 {
                    bits |= 0x8000_0000;
                }
                out[i * 4..i * 4 + 4].copy_from_slice(&bits.to_le_bytes());
            }
            out
        }
        other => panic!("unknown dtype {other}"),
    }
}

fn lcg_ids(seed: u32, n: usize, modulo: u32, repeat_tail: usize) -> Vec<u32> {
    let mut x = seed;
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        x = lcg_next(x);
        ids.push(x % modulo);
    }
    if repeat_tail > 0 && n >= 2 * repeat_tail {
        let start = n - repeat_tail;
        let pattern_at = n - 2 * repeat_tail;
        for i in 0..repeat_tail {
            ids[start + i] = ids[pattern_at + i];
        }
    }
    ids
}

fn history_bytes(ids: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * 4);
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

fn apply_patch(row: &mut [u8], patch: &serde_json::Value) {
    if patch.is_null() {
        return;
    }
    let index = patch["index"].as_u64().unwrap() as usize;
    let bits = patch["f16_bits"].as_u64().unwrap() as u16;
    row[index * 2..index * 2 + 2].copy_from_slice(&bits.to_le_bytes());
}

fn addr(buf: &DeviceBuffer<u8>) -> *mut c_void {
    buf.ptr().as_u64() as *mut c_void
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0xf)]));
    }
    out
}

/// blake2b-256. The kernel crate has no blake2 dependency. A logit row is an
/// exact multiple of the 128-byte block, so a full block stays buffered until
/// finalize and is compressed once, as the last block.
fn blake2b_256(data: &[u8]) -> String {
    const IV: [u64; 8] = [
        0x6a09_e667_f3bc_c908,
        0xbb67_ae85_84ca_a73b,
        0x3c6e_f372_fe94_f82b,
        0xa54f_f53a_5f1d_36f1,
        0x510e_527f_ade6_82d1,
        0x9b05_688c_2b3e_6c1f,
        0x1f83_d9ab_fb41_bd6b,
        0x5be0_cd19_137e_2179,
    ];
    const SIGMA: [[usize; 16]; 12] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
        [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
        [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
        [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
        [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
        [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
        [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
        [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
        [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    ];

    fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(32);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(24);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(63);
    }

    fn compress(h: &mut [u64; 8], block: &[u8], t0: u64, t1: u64, last: bool) {
        let mut m = [0u64; 16];
        for (i, word) in m.iter_mut().enumerate() {
            let mut bytes = [0u8; 8];
            bytes.copy_from_slice(&block[i * 8..i * 8 + 8]);
            *word = u64::from_le_bytes(bytes);
        }
        let mut v = [0u64; 16];
        v[..8].copy_from_slice(h);
        v[8..].copy_from_slice(&IV);
        v[12] ^= t0;
        v[13] ^= t1;
        if last {
            v[14] = !v[14];
        }
        for sigma in SIGMA {
            g(&mut v, 0, 4, 8, 12, m[sigma[0]], m[sigma[1]]);
            g(&mut v, 1, 5, 9, 13, m[sigma[2]], m[sigma[3]]);
            g(&mut v, 2, 6, 10, 14, m[sigma[4]], m[sigma[5]]);
            g(&mut v, 3, 7, 11, 15, m[sigma[6]], m[sigma[7]]);
            g(&mut v, 0, 5, 10, 15, m[sigma[8]], m[sigma[9]]);
            g(&mut v, 1, 6, 11, 12, m[sigma[10]], m[sigma[11]]);
            g(&mut v, 2, 7, 8, 13, m[sigma[12]], m[sigma[13]]);
            g(&mut v, 3, 4, 9, 14, m[sigma[14]], m[sigma[15]]);
        }
        for i in 0..8 {
            h[i] ^= v[i] ^ v[i + 8];
        }
    }

    let mut h = IV;
    h[0] ^= 0x0101_0020;
    let mut buf = [0u8; 128];
    let mut buflen = 0usize;
    let mut t0 = 0u64;
    let mut t1 = 0u64;
    let mut rest = data;
    while !rest.is_empty() {
        if buflen == 128 {
            let (next, carry) = t0.overflowing_add(128);
            t0 = next;
            if carry {
                t1 += 1;
            }
            compress(&mut h, &buf, t0, t1, false);
            buflen = 0;
        }
        let n = (128 - buflen).min(rest.len());
        buf[buflen..buflen + n].copy_from_slice(&rest[..n]);
        buflen += n;
        rest = &rest[n..];
    }
    let (next, carry) = t0.overflowing_add(buflen as u64);
    t0 = next;
    if carry {
        t1 += 1;
    }
    for byte in buf.iter_mut().skip(buflen) {
        *byte = 0;
    }
    compress(&mut h, &buf, t0, t1, true);
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate().take(4) {
        out[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    hex_encode(&out)
}

fn preview(bytes: &[u8], dtype: &str) -> String {
    if dtype == "f16" {
        let n = (bytes.len() / 2).min(8);
        let mut parts = Vec::new();
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
            parts.push(format!("0x{bits:04x}"));
        }
        parts.join(" ")
    } else {
        let n = (bytes.len() / 4).min(8);
        let mut parts = Vec::new();
        for i in 0..n {
            let bits = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            parts.push(format!("{}", f32::from_bits(bits)));
        }
        parts.join(" ")
    }
}

fn u32_at(value: &serde_json::Value, key: &str) -> u32 {
    value[key]
        .as_u64()
        .unwrap_or_else(|| panic!("missing {key}")) as u32
}

fn i32_at(value: &serde_json::Value, key: &str) -> i32 {
    value[key]
        .as_i64()
        .unwrap_or_else(|| panic!("missing {key}")) as i32
}

fn f32_at(value: &serde_json::Value, key: &str) -> f32 {
    f32::from_bits(u32_at(value, key))
}

#[test]
fn blake2b_256_matches_hashlib() {
    assert_eq!(
        blake2b_256(b""),
        "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8"
    );
    assert_eq!(
        blake2b_256(b"abc"),
        "bddd813c634239723171ef3fee98579b94964e3bb1cb3e427262c8c068d52319"
    );
    assert_eq!(
        blake2b_256(b"berth-sample"),
        "03bb55c146d39daee08be41f991b62baa74dd8249a76ae008836dc2b33a0790f"
    );
    // Exact multiples of the block. A logit row is one of these.
    assert_eq!(
        blake2b_256(&[b'x'; 128]),
        "164ffb7089bae6f5a62fb0795e751dc9e88eac92e1a5b2fafe93a25abf2d9c3b"
    );
    assert_eq!(
        blake2b_256(&[b'x'; 256]),
        "cb9d468022b930d52da4d9b0226ef9c997e0acc119eeb08afefb8219bc53972b"
    );
}

#[test]
fn null_pointers_do_not_call_cuda() {
    let err = unsafe {
        sample::apply_rep_pens(
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            0,
            8,
            true,
            1.1,
            4,
            0,
            std::ptr::null_mut(),
        )
    }
    .expect_err("null");
    assert!(err.to_string().contains("null"), "{err}");
    assert!(
        sample::last_launch().is_empty(),
        "{}",
        sample::last_launch()
    );

    let err = unsafe {
        sample::dry_penalty(
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
            8,
            true,
            0.2,
            1.75,
            6,
            16,
            158,
            2048,
            std::ptr::null_mut(),
        )
    }
    .expect_err("null dry");
    assert!(err.to_string().contains("null"), "{err}");

    let err = unsafe {
        sample::gumbel_f32(
            std::ptr::null(),
            std::ptr::null_mut(),
            4,
            1,
            std::ptr::null_mut(),
        )
    }
    .expect_err("null gumbel");
    assert!(err.to_string().contains("null"), "{err}");

    let err = unsafe {
        sample::fused(
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            8,
            8,
            true,
            1.0,
            0,
            1,
            std::ptr::null_mut(),
        )
    }
    .expect_err("null fused");
    assert!(err.to_string().contains("null"), "{err}");

    let err = unsafe {
        sample::fused(
            std::ptr::dangling(),
            std::ptr::dangling_mut(),
            std::ptr::dangling_mut(),
            8,
            8,
            true,
            1.0,
            0,
            3,
            std::ptr::null_mut(),
        )
    }
    .expect_err("mode");
    assert!(err.to_string().contains("mode"), "{err}");
    assert!(sample::last_launch().is_empty());
}

#[test]
fn kernels_match_the_fork() {
    if std::env::var("BERTH_GPU_TESTS").ok().as_deref() != Some("1") {
        eprintln!("skipping sample kernels: set BERTH_GPU_TESTS=1");
        return;
    }
    if serve_listening() {
        eprintln!("skipping sample kernels: port 8889 is in use or ss failed");
        return;
    }
    let meta = load_meta();
    let vocab = meta["vocab"].as_u64().unwrap() as usize;
    let ctx = Context::new().unwrap();
    let stream = ctx.stream().unwrap();
    let raw = stream.handle().as_ptr();
    for case in meta["kernel_cases"].as_array().unwrap() {
        run_case(&stream, raw, vocab, case);
    }
}

fn run_case(stream: &Stream, raw: *mut c_void, vocab: usize, case: &serde_json::Value) {
    let name = case["name"].as_str().unwrap();
    let op = case["op"].as_str().unwrap();
    let dtype = case["dtype"].as_str().unwrap();
    let mut row = lcg_row(u32_at(case, "seed"), vocab, dtype);
    apply_patch(&mut row, &case["patch"]);
    let input_hash = blake2b_256(&row);
    assert_eq!(
        input_hash,
        case["input_blake2b256"].as_str().unwrap(),
        "{name}: input LCG drifted"
    );

    let history = if case.get("history_len").is_some() {
        let ids = lcg_ids(
            u32_at(case, "history_seed"),
            u32_at(case, "history_len") as usize,
            u32_at(case, "history_mod"),
            u32_at(case, "repeat_tail") as usize,
        );
        let hash = blake2b_256(&history_bytes(&ids));
        assert_eq!(
            hash,
            case["history_blake2b256"].as_str().unwrap(),
            "{name}: history LCG drifted"
        );
        ids
    } else {
        Vec::new()
    };
    let past_i64: Vec<i64> = history.iter().copied().map(i64::from).collect();
    let past = if past_i64.is_empty() {
        // The kernel requires a non-null pointer even when past_len is 0.
        DeviceBuffer::from_host(stream, &[0i64]).unwrap()
    } else {
        DeviceBuffer::from_host(stream, &past_i64).unwrap()
    };
    let past_ptr = past.ptr().as_u64() as *const c_void;
    let past_len = history.len() as i32;

    let inplace = case["inplace"].as_bool().unwrap();
    let input = DeviceBuffer::from_host(stream, &row).unwrap();
    // Moving a DeviceBuffer keeps the device address, so the pointer taken
    // before the move stays valid.
    let mut separate: Option<DeviceBuffer<u8>> = None;
    let out_ptr = if inplace {
        addr(&input)
    } else if op == "fused" {
        let buf = DeviceBuffer::<u8>::zeroed(stream, 8).unwrap();
        let ptr = addr(&buf);
        separate = Some(buf);
        ptr
    } else {
        let bytes = if op == "gumbel_f16" {
            vocab * 2
        } else {
            vocab * 4
        };
        let buf = DeviceBuffer::<u8>::zeroed(stream, bytes).unwrap();
        let ptr = addr(&buf);
        separate = Some(buf);
        ptr
    };
    let in_ptr = addr(&input);
    let params = &case["params"];

    let breakers = if op == "dry" {
        let ids = case["breakers"].as_array().unwrap();
        if ids.is_empty() {
            None
        } else {
            let mut mask = vec![0u8; vocab];
            for id in ids {
                let id = id.as_u64().unwrap() as usize;
                if id < vocab {
                    mask[id] = 1;
                }
            }
            Some(DeviceBuffer::from_host(stream, &mask).unwrap())
        }
    } else {
        None
    };
    let scratch = if op == "dry" {
        Some(DeviceBuffer::<u8>::zeroed(stream, (vocab + 1) * 4).unwrap())
    } else {
        None
    };
    let workspace = if op == "fused" {
        Some(DeviceBuffer::<u8>::zeroed(stream, 128 * 3 * 4).unwrap())
    } else {
        None
    };

    let result = unsafe {
        match op {
            "rep" => sample::apply_rep_pens(
                in_ptr,
                out_ptr,
                past_ptr,
                past_len,
                vocab as i32,
                dtype == "f16",
                f32_at(params, "rep_p_bits"),
                i32_at(params, "sustain"),
                i32_at(params, "decay"),
                raw,
            ),
            "pres" => sample::apply_pres_freq_pens(
                in_ptr,
                out_ptr,
                past_ptr,
                past_len,
                vocab as i32,
                dtype == "f16",
                f32_at(params, "pres_p_bits"),
                f32_at(params, "freq_p_bits"),
                i32_at(params, "sustain"),
                i32_at(params, "decay"),
                raw,
            ),
            "dry" => sample::dry_penalty(
                in_ptr,
                out_ptr,
                past_ptr,
                breakers.as_ref().map(addr).unwrap_or(std::ptr::null_mut()) as *const c_void,
                addr(scratch.as_ref().unwrap()),
                past_len,
                vocab as i32,
                dtype == "f16",
                f32_at(params, "multiplier_bits"),
                f32_at(params, "base_bits"),
                i32_at(params, "allowed_length"),
                i32_at(params, "range"),
                i32_at(params, "max_exponent"),
                i32_at(params, "match_cap"),
                raw,
            ),
            "gumbel_f16" => {
                sample::gumbel_f16(in_ptr, out_ptr, vocab as i32, u32_at(params, "random"), raw)
            }
            "gumbel_f32" => {
                sample::gumbel_f32(in_ptr, out_ptr, vocab as i32, u32_at(params, "random"), raw)
            }
            "fused" => sample::fused(
                in_ptr,
                out_ptr,
                addr(workspace.as_ref().unwrap()),
                vocab as i32,
                i32_at(params, "size"),
                dtype == "f16",
                f32_at(params, "inv_temp_bits"),
                u32_at(params, "random"),
                i32_at(params, "mode"),
                raw,
            ),
            other => panic!("{name}: unknown op {other}"),
        }
    };
    result.unwrap_or_else(|err| panic!("{name}: {err}"));
    let launch = sample::last_launch();
    for needle in case["launch_contains"].as_array().unwrap() {
        let needle = needle.as_str().unwrap();
        assert!(
            launch.contains(needle),
            "{name}: launch `{launch}` missing `{needle}`"
        );
    }

    let src: &DeviceBuffer<u8> = if inplace {
        &input
    } else {
        separate.as_ref().unwrap()
    };
    if op == "fused" {
        let mut token = [0u8; 8];
        src.copy_to_host(stream, &mut token).unwrap();
        let got = u64::from_le_bytes(token);
        let expect = case["token"].as_u64().unwrap();
        assert_eq!(got, expect, "{name}: token {got} != {expect}");
    } else {
        let mut got = vec![0u8; src.num_bytes()];
        src.copy_to_host(stream, &mut got).unwrap();
        let hash = blake2b_256(&got);
        let expect = case["output_blake2b256"].as_str().unwrap();
        assert_eq!(
            hash,
            expect,
            "{name}: output mismatch, first values {}",
            preview(&got, if op == "gumbel_f16" { "f16" } else { "f32" })
        );
    }
}
