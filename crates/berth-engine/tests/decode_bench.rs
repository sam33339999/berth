//! Eager decode speed after one warm-up step.
//!
//! Ignored so a normal `cargo test` does not time the GPU. Run:
//!
//! ```text
//! BERTH_GPU_TESTS=1 cargo test -p berth-engine --test decode_bench -- --ignored --nocapture
//! ```
//!
//! The fork's graphed decode, from `docs/kernels.md`, is 25.81 ms/token
//! (about 38.7 tokens/s): 128 `cudaGraphLaunch` calls per token, measured
//! on this 3090 with the service stopped.

mod common;

use berth_engine::model::Model;
use berth_engine::Gpu;
use berth_exl3::Checkpoint;
use berth_plan::{build_plan, from_exl3::model_shape};

/// Same 4096-token cache as `decode_ref`. The timed steps stay inside it.
const MAX_TOKENS: u32 = 4096;
const TIMED_STEPS: u32 = 64;

#[test]
#[ignore = "times 64 eager decode steps; needs BERTH_GPU_TESTS=1"]
fn eager_decode_64() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let _slot = match common::full_model_slot() {
        Ok(slot) => slot,
        Err(why) => return eprintln!("skipping: {why}"),
    };

    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let plan = build_plan(&shape).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut model = Model::load(&gpu, &ckpt, &plan, MAX_TOKENS).unwrap();
    let token = common::reference_token();

    // Fills the mgemm / lm_head autotune memory cache and the one-time
    // int8 workspace allocation. The position-0 token stays in the cache;
    // the timed steps continue from position 1.
    model.decode_step(&gpu, token, 0).unwrap();
    model.synchronize(&gpu).unwrap();

    let start = gpu.ctx.event().unwrap();
    let end = gpu.ctx.event().unwrap();
    start.record(&gpu.stream).unwrap();
    for step in 0..TIMED_STEPS {
        model.decode_step(&gpu, token, 1 + step).unwrap();
    }
    end.record(&gpu.stream).unwrap();
    let total_ms = start.elapsed_ms(&end).unwrap();
    let ms = total_ms / TIMED_STEPS as f32;
    let per_sec = 1000.0 / ms;
    // Fork (docs/kernels.md): 25.81 ms/token, about 38.7 tokens/s, graph replay.
    eprintln!(
        "eager decode: {ms:.3} ms/token, {per_sec:.2} tokens/s ({TIMED_STEPS} steps, {total_ms:.1} ms)"
    );
    assert!(ms.is_finite() && ms > 0.0, "ms/token = {ms}");
}
