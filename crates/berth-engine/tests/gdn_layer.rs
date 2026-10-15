//! Layer 0 Gated DeltaNet against `testdata/decode_ref`, bit for bit. GPU only.
//!
//! Position 0: the layer's conv and recurrent state start at zero, which is
//! the reference cache before the token.

mod common;

use berth_engine::gdn::GdnLayer;
use berth_engine::Gpu;
use berth_exl3::Checkpoint;
use berth_plan::from_exl3::model_shape;

fn serve_listening() -> bool {
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
fn gdn_layer0_matches_the_reference() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    if serve_listening() {
        return eprintln!("skipping: port 8889 is listening");
    }

    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut layer = GdnLayer::load(&gpu, &ckpt, &shape, 0).unwrap();

    let input_bytes = common::reference("block_00_mixer_in");
    let want = common::reference("block_00_mixer");
    assert_eq!(input_bytes.len() * 2, want.len(), "fp16 input, fp32 output");
    let input = common::upload(&gpu, &input_bytes);
    let mut output = gpu.zeroed(want.len()).unwrap();
    layer.forward(&gpu, &input, &mut output).unwrap();
    let got = common::download(&gpu, &output);
    let n = common::differing(&got, &want, 4);
    eprintln!(
        "block_00_mixer differing elements: {n} of {}",
        want.len() / 4
    );
    eprintln!("mgemm: {}", berth_kernels::ops::last_launch());
    eprintln!("gdn: {}", berth_kernels::gdn::last_launch());
    eprintln!("out_proj: {}", berth_kernels::last_launch());
    if n != 0 {
        let index = got
            .chunks(4)
            .zip(want.chunks(4))
            .position(|(got_elem, want_elem)| got_elem != want_elem)
            .unwrap();
        let got_f = f32::from_le_bytes(got[index * 4..index * 4 + 4].try_into().unwrap());
        let want_f = f32::from_le_bytes(want[index * 4..index * 4 + 4].try_into().unwrap());
        panic!("block_00_mixer: {n} elements differ; first at {index}: got {got_f} want {want_f}");
    }
}
