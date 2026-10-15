//! Shared helpers for the engine's GPU tests against `testdata/decode_ref/`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use berth_cuda::{ByteBuf, DeviceBuffer};
use berth_engine::Gpu;

/// `BERTH_GPU_TESTS=1` and the checkpoint is present; otherwise the reason to skip.
pub fn gpu_ready() -> Result<String, String> {
    if std::env::var("BERTH_GPU_TESTS").as_deref() != Ok("1") {
        return Err("set BERTH_GPU_TESTS=1".into());
    }
    let dir = std::env::var("BERTH_27B_DIR")
        .unwrap_or_else(|_| "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3".to_string());
    if !Path::new(&dir).is_dir() {
        return Err(format!("checkpoint directory missing: {dir}"));
    }
    Ok(dir)
}

pub fn decode_ref_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/decode_ref")
}

/// Bytes of one `decode_ref` tensor, by its `meta.json` name (e.g. `block_00_mixer`).
pub fn reference(name: &str) -> Vec<u8> {
    std::fs::read(decode_ref_dir().join(format!("{name}.bin")))
        .unwrap_or_else(|err| panic!("decode_ref {name}: {err}"))
}

/// The token id the reference decoded.
pub fn reference_token() -> u32 {
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(decode_ref_dir().join("meta.json")).unwrap())
            .unwrap();
    meta["token"].as_u64().unwrap() as u32
}

pub fn upload(gpu: &Gpu, bytes: &[u8]) -> ByteBuf {
    DeviceBuffer::from_host(&gpu.stream, bytes).unwrap()
}

pub fn download(gpu: &Gpu, buf: &ByteBuf) -> Vec<u8> {
    gpu.stream.synchronize().unwrap();
    let mut out = vec![0u8; buf.num_bytes()];
    buf.copy_to_host(&gpu.stream, &mut out).unwrap();
    out
}

/// Number of `elem`-byte elements that differ.
pub fn differing(got: &[u8], want: &[u8], elem: usize) -> usize {
    assert_eq!(got.len(), want.len(), "byte length");
    got.chunks(elem)
        .zip(want.chunks(elem))
        .filter(|(a, b)| a != b)
        .count()
}
