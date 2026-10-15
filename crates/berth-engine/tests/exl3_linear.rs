//! `Exl3Linear` against `testdata/exl3_ref`, bit for bit. GPU only.

mod common;

use berth_engine::{Exl3Linear, Gpu};
use berth_exl3::Checkpoint;
use berth_kernels::OutDtype;

#[test]
fn exl3_linear_matches_the_reference() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    let refs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/exl3_ref");
    let ckpt = Checkpoint::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();
    for (name, key, out, elem) in [
        (
            "l0_down_m1",
            "model.language_model.layers.0.mlp.down_proj",
            OutDtype::F32,
            4,
        ),
        ("lm_head_m1", "lm_head", OutDtype::F16, 2),
    ] {
        let mut linear = Exl3Linear::load(&gpu, &ckpt, key).unwrap();
        let x = std::fs::read(refs.join(format!("{name}.x.bin"))).unwrap();
        let want = std::fs::read(refs.join(format!("{name}.y.bin"))).unwrap();
        let a = common::upload(&gpu, &x);
        let mut c = gpu.zeroed(linear.size_n as usize * elem).unwrap();
        linear.forward(&gpu, &a, &mut c, out).unwrap();
        let got = common::download(&gpu, &c);
        assert_eq!(common::differing(&got, &want, elem), 0, "{name}");
    }
}
