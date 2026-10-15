//! Full `Model::load` plus `decode_step` against `testdata/decode_ref`.
//!
//! Loads all 64 layers (about 16 GB of VRAM). Runs only with
//! `BERTH_GPU_TESTS=1`:
//!
//! ```text
//! BERTH_GPU_TESTS=1 cargo test -p berth-engine --test decode_ref -- --nocapture
//! ```
//!
//! The reference cache is 4096 tokens (`testdata/decode_ref/meta.json`).
//! The step is the reference token at position 0.

mod common;

use berth_engine::model::Model;
use berth_engine::Gpu;
use berth_exl3::Checkpoint;
use berth_plan::{build_plan, from_exl3::model_shape};

/// Attention cache length recorded for the reference decode.
const MAX_TOKENS: u32 = 4096;

#[test]
fn decode_step_matches_decode_ref() {
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
    let layers = model.num_layers();
    let token = common::reference_token();
    // Download before reading the block residuals: `decode_step` borrows `model`.
    let got_logits = {
        let logits = model.decode_step(&gpu, token, 0).unwrap();
        common::download(&gpu, logits)
    };

    for index in 0..layers {
        let name = format!("block_{index:02}");
        let got = model.block_residual(u32::try_from(index).unwrap()).unwrap();
        let want = common::reference(&name);
        let diff = common::differing(got, &want, 4);
        let n = want.len() / 4;
        eprintln!("{name}: {diff} of {n} elements differ");
        if diff != 0 {
            eprintln!("first differing block: {name} ({diff} of {n} elements)");
            panic!("first differing block: {name}");
        }
    }

    let want_logits = common::reference("logits");
    let diff = common::differing(&got_logits, &want_logits, 2);
    let n = want_logits.len() / 2;
    eprintln!("logits: {diff} of {n} elements differ");
    assert_eq!(diff, 0, "logits");
}
