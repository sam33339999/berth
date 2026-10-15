//! `MlpLayer` against `testdata/decode_ref`, bit for bit. GPU only.
//!
//! Layers 0 and 3: fp16 `block_NN_mlp_in` in, fp32 `block_NN_mlp` out.

mod common;

use berth_engine::mlp::MlpLayer;
use berth_engine::Gpu;
use berth_exl3::Checkpoint;

/// `:8889` is the serve. Same probe as the task: `ss -ltn | grep -q ':8889 '`.
fn serve_blocks_gpu() -> bool {
    let output = match std::process::Command::new("ss").args(["-ltn"]).output() {
        Ok(output) => output,
        Err(_) => return true,
    };
    if !output.status.success() {
        return true;
    }
    String::from_utf8_lossy(&output.stdout).contains(":8889 ")
}

#[test]
fn mlp_layer_matches_the_reference() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    if serve_blocks_gpu() {
        return eprintln!("skipping: port 8889 is in use or ss failed");
    }

    let ckpt = Checkpoint::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();
    for layer in [0_u32, 3] {
        let mut mlp = MlpLayer::load(&gpu, &ckpt, layer).unwrap();
        let stem = format!("block_{layer:02}");
        let input = common::reference(&format!("{stem}_mlp_in"));
        let want = common::reference(&format!("{stem}_mlp"));
        let x = common::upload(&gpu, &input);
        let mut y = gpu.zeroed(want.len()).unwrap();
        mlp.forward(&gpu, &x, &mut y).unwrap();
        let got = common::download(&gpu, &y);
        let n = common::differing(&got, &want, 4);
        if n != 0 {
            let mut shown = 0;
            for (i, (a, b)) in got.chunks(4).zip(want.chunks(4)).enumerate() {
                if a != b {
                    let got_v = f32::from_le_bytes(a.try_into().unwrap());
                    let want_v = f32::from_le_bytes(b.try_into().unwrap());
                    eprintln!("  [{i}] got {got_v:e} want {want_v:e}");
                    shown += 1;
                    if shown == 8 {
                        break;
                    }
                }
            }
        }
        eprintln!(
            "{stem}_mlp: {n} of {} elements differ; down {}; silu {}",
            want.len() / 4,
            berth_kernels::last_launch(),
            berth_kernels::ops::last_launch(),
        );
        assert_eq!(n, 0, "{stem}_mlp");
        // Drop this layer's weights before loading the next one.
        drop(mlp);
        drop(x);
        drop(y);
    }
}
