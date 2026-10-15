//! Layer 0 Gated DeltaNet prefill against `/home/sam/berth/target/prefill_layers`.
//!
//! Prompt 100 is one chunk of 99 rows. Prompt 1500 is 1280 rows, then 219
//! rows on the same conv and recurrent state. A decode step after the
//! 100-token chunk checks that the state handoff is finite.

mod common;

use std::path::Path;

use berth_engine::device::RowScratch;
use berth_engine::gdn::{GdnLayer, GdnPrefillScratch};
use berth_engine::Gpu;
use berth_exl3::Checkpoint;
use berth_plan::from_exl3::model_shape;

const PREFILL_DIR: &str = "/home/sam/berth/target/prefill_layers";

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

fn prefill_tensor(name: &str) -> Vec<u8> {
    let path = Path::new(PREFILL_DIR).join(format!("{name}.bin"));
    std::fs::read(&path).unwrap_or_else(|err| panic!("prefill tensor {}: {err}", path.display()))
}

#[test]
fn gdn_layer0_prefill_matches_the_reference() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    if serve_listening() {
        return eprintln!("skipping: port 8889 is listening");
    }
    if !Path::new(PREFILL_DIR).is_dir() {
        return eprintln!("skipping: {PREFILL_DIR} is missing");
    }

    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let hidden = shape.hidden_size;
    let v_dim = shape.gdn.num_v_heads * shape.gdn.value_head_dim;
    let gpu = Gpu::new().unwrap();
    let mut layer = GdnLayer::load(&gpu, &ckpt, &shape, 0).unwrap();
    let mut scratch = GdnPrefillScratch::new(&gpu, &shape, 2048).unwrap();
    let mut rows_scratch = RowScratch::new(&gpu, 2048, v_dim, hidden).unwrap();

    compare(
        &gpu,
        &mut layer,
        &mut rows_scratch,
        &mut scratch,
        "100_c0_b00",
        99,
    );

    let mixer_in = prefill_tensor("100_c0_b00_mixer_in");
    let row_bytes = hidden as usize * 2;
    let one = common::upload(&gpu, &mixer_in[..row_bytes]);
    let mut decoded = gpu.zeroed(hidden as usize * 4).unwrap();
    layer.forward(&gpu, &one, &mut decoded).unwrap();
    let got = common::download(&gpu, &decoded);
    let values: Vec<f32> = got
        .chunks_exact(4)
        .map(|elem| f32::from_le_bytes(elem.try_into().unwrap()))
        .collect();
    let non_finite = values.iter().filter(|v| !v.is_finite()).count();
    eprintln!(
        "decode after 100-token prefill: {non_finite} non-finite of {}",
        values.len()
    );
    assert_eq!(non_finite, 0, "decode handoff produced a non-finite value");
    assert!(
        values.iter().any(|v| *v != 0.0),
        "decode handoff wrote zeros"
    );

    layer.reset_state(&gpu).unwrap();
    compare(
        &gpu,
        &mut layer,
        &mut rows_scratch,
        &mut scratch,
        "1500_c0_b00",
        1280,
    );
    compare(
        &gpu,
        &mut layer,
        &mut rows_scratch,
        &mut scratch,
        "1500_c1_b00",
        219,
    );
}

fn compare(
    gpu: &Gpu,
    layer: &mut GdnLayer,
    rows_scratch: &mut RowScratch,
    scratch: &mut GdnPrefillScratch,
    stem: &str,
    rows: u32,
) {
    let input_bytes = prefill_tensor(&format!("{stem}_mixer_in"));
    let want = prefill_tensor(&format!("{stem}_mixer"));
    let hidden = 5120usize;
    let n = rows as usize;
    assert_eq!(input_bytes.len(), n * hidden * 2, "{stem} mixer_in bytes");
    assert_eq!(want.len(), n * hidden * 4, "{stem} mixer bytes");
    let input = common::upload(gpu, &input_bytes);
    let mut output = gpu.zeroed(want.len()).unwrap();
    layer
        .prefill(gpu, rows_scratch, scratch, &input, &mut output, rows)
        .unwrap();
    let got = common::download(gpu, &output);
    let differing = common::differing(&got, &want, 4);
    eprintln!(
        "{stem} mixer differing elements: {differing} of {}",
        want.len() / 4
    );
    eprintln!("exl3: {}", berth_kernels::prefill_lin::last_launch());
    if differing != 0 {
        let index = got
            .chunks_exact(4)
            .zip(want.chunks_exact(4))
            .position(|(got_elem, want_elem)| got_elem != want_elem)
            .unwrap();
        let got_f = f32::from_le_bytes(got[index * 4..index * 4 + 4].try_into().unwrap());
        let want_f = f32::from_le_bytes(want[index * 4..index * 4 + 4].try_into().unwrap());
        panic!("{stem} mixer: {differing} elements differ; first at {index}: got {got_f} want {want_f}");
    }
}
