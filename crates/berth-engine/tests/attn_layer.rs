//! Layer 3 full attention against `testdata/decode_ref/block_03_mixer`. GPU only.
//!
//! The reference dump's generator pads the block table to a multiple of 16
//! (`iterate_gen` in `generator/generator.py`). One token on a 4096-token
//! cache is 16 pages, and `split_config` keys off that width. Position 0 and
//! the follow-up token both land in page 0.

mod common;

use berth_engine::attn::AttnLayer;
use berth_engine::Gpu;
use berth_exl3::Checkpoint;
use berth_kernels::attn::PagedDecode;
use berth_plan::from_exl3::model_shape;

#[test]
fn layer3_matches_block_03_mixer() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    if let Some(why) = port_blocked() {
        return eprintln!("skipping: {why}");
    }
    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let gpu = Gpu::new().unwrap();
    let cc = gpu.ctx.compute_capability().unwrap();
    if gpu.sm_count != 82 || (cc.major, cc.minor) != (8, 6) {
        return eprintln!(
            "skipping: mgemm cache is for sm_86 with 82 SMs (got sm_{}{}, {} SMs); refusing to retune",
            cc.major, cc.minor, gpu.sm_count
        );
    }

    let cubins =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins");
    let kernels = PagedDecode::load(&gpu.ctx, &cubins).unwrap();
    // 4096 tokens → 16 pages, the padded width of this dump.
    let mut layer = AttnLayer::load(&gpu, &ckpt, &shape, 3, 4096).unwrap();
    let input = common::upload(&gpu, &common::reference("block_03_mixer_in"));
    let mut output = gpu.zeroed(shape.hidden_size as usize * 4).unwrap();
    let stamp = autotune_stamp();

    layer
        .forward(&gpu, &kernels, &input, &mut output, 0)
        .unwrap();
    let got = common::download(&gpu, &output);
    let want = common::reference("block_03_mixer");
    let diff = common::differing(&got, &want, 4);
    eprintln!(
        "layer 3 position 0: {diff} of {} elements differ",
        want.len() / 4
    );
    if diff != 0 {
        let mut shown = 0;
        for (index, (got_b, want_b)) in got.chunks(4).zip(want.chunks(4)).enumerate() {
            if got_b == want_b {
                continue;
            }
            let got_f = f32::from_le_bytes(got_b.try_into().unwrap());
            let want_f = f32::from_le_bytes(want_b.try_into().unwrap());
            eprintln!("  [{index}] got {got_f} ({got_b:02x?}) want {want_f} ({want_b:02x?})");
            shown += 1;
            if shown == 8 {
                break;
            }
        }
    }
    assert_eq!(diff, 0, "block_03_mixer");

    // Same input at position 1. The dump has no second token.
    layer
        .forward(&gpu, &kernels, &input, &mut output, 1)
        .unwrap();
    let second = common::download(&gpu, &output);
    let finite = second
        .chunks(4)
        .all(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()).is_finite());
    eprintln!("layer 3 position 1: finite={finite}; no reference tensor for the second token");
    assert!(finite, "position 1 output has a non-finite value");
    if stamp != autotune_stamp() {
        eprintln!("mgemm autotune cache changed during the layer test");
    }
}

fn port_blocked() -> Option<String> {
    let output = match std::process::Command::new("ss").args(["-ltn"]).output() {
        Ok(output) => output,
        Err(err) => return Some(format!("ss failed ({err}); not using the GPU")),
    };
    if !output.status.success() {
        return Some(format!("ss exited {}; not using the GPU", output.status));
    }
    let text = String::from_utf8_lossy(&output.stdout);
    if text.lines().any(|line| line.contains(":8889 ")) {
        return Some("port 8889 is listening".into());
    }
    None
}

fn autotune_stamp() -> Option<(u64, std::time::SystemTime)> {
    let path = std::env::var("EXLLAMAV3_TUNE_CACHE").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.cache/exllamav3/autotune/coop_autotune_v1.bin")
    });
    let meta = std::fs::metadata(path).ok()?;
    Some((
        meta.len(),
        meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH),
    ))
}
