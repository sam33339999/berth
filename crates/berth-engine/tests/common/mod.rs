//! Shared helpers for the engine's GPU tests against `testdata/decode_ref/`.
#![allow(dead_code)]

use std::fs::File;

use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use berth_cuda::{ByteBuf, DeviceBuffer};
use berth_engine::Gpu;

/// Exclusive lock so two full-model tests do not load the 27B together.
const GPU_LOCK: &str = "/tmp/berth-engine-gpu.lock";
/// `ss -ltn` shows this when the Python service is up.
const SERVE_PORT: &str = ":8889 ";
/// Task READY loads the Rust 27B only when the card is under this.
const MODEL_MEMORY_MIB: u32 = 4000;
/// How long to wait for that, matching the task's 20 minute cap.
const MEMORY_WAIT: Duration = Duration::from_secs(20 * 60);
const MEMORY_POLL: Duration = Duration::from_secs(60);

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

/// Held for the whole full-model load. Drop releases the lock.
pub struct FullModelSlot {
    file: File,
}

impl Drop for FullModelSlot {
    fn drop(&mut self) {
        // LOCK_UN. The process exit also drops the lock; this covers the
        // test returning while the binary keeps running other tests.
        unsafe {
            flock(self.file.as_raw_fd(), 8);
        }
    }
}

/// Port 8889 free, GPU memory under 4000 MiB, and an exclusive lock.
///
/// Waits up to 20 minutes when the card is busy. Returns the reason to skip
/// when the service is listening or the wait runs out.
pub fn full_model_slot() -> Result<FullModelSlot, String> {
    if serve_listening()? {
        return Err("port 8889 is in use".into());
    }
    let file = File::options()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(GPU_LOCK)
        .map_err(|err| format!("opening {GPU_LOCK}: {err}"))?;
    // LOCK_EX. Another engine test loading the 27B blocks here.
    let rc = unsafe { flock(file.as_raw_fd(), 2) };
    if rc != 0 {
        return Err(format!("flock {GPU_LOCK} failed ({rc})"));
    }
    let deadline = Instant::now() + MEMORY_WAIT;
    loop {
        let used = memory_used_mib()?;
        if used < MODEL_MEMORY_MIB {
            return Ok(FullModelSlot { file });
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "GPU memory still {used} MiB after {} min",
                MEMORY_WAIT.as_secs() / 60
            ));
        }
        eprintln!("GPU memory {used} MiB; waiting 60s");
        thread::sleep(MEMORY_POLL);
    }
}

fn serve_listening() -> Result<bool, String> {
    let output = Command::new("ss")
        .args(["-ltn"])
        .output()
        .map_err(|err| format!("ss failed ({err}); not using the GPU"))?;
    if !output.status.success() {
        return Err(format!("ss exited {}; not using the GPU", output.status));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Ok(text.contains(SERVE_PORT))
}

fn memory_used_mib() -> Result<u32, String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader"])
        .output()
        .map_err(|err| format!("nvidia-smi failed ({err})"))?;
    if !output.status.success() {
        return Err(format!("nvidia-smi exited {}", output.status));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next().unwrap_or("").trim();
    let digits: String = line.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits
        .parse()
        .map_err(|_| format!("nvidia-smi memory `{line}` has no leading integer"))
}

extern "C" {
    fn flock(fd: i32, op: i32) -> i32;
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
