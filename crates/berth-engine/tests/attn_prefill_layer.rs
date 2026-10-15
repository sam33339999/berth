//! Layer 3 full-attention prefill against `/home/sam/berth/target/prefill_layers`.
//!
//! Prompt 100 is one chunk of 99 rows at position 0 (`ids[:-1]`). Prompt 1500
//! is 1280 rows at 0, then 219 rows at 1280, on the same cache. Mixer output
//! is fp32 `[rows, hidden]`. The dump's leading 1 is the batch.

mod common;

use std::path::Path;

use berth_engine::attn::{AttnLayer, AttnPrefillScratch};
use berth_engine::device::RowScratch;
use berth_engine::prefill::MAX_CHUNK_SIZE;
use berth_engine::Gpu;
use berth_exl3::Checkpoint;
use berth_plan::from_exl3::model_shape;

const LAYERS: &str = "/home/sam/berth/target/prefill_layers";

#[test]
fn layer3_prefill_matches_the_fork() {
    if std::env::var("BERTH_GPU_TESTS").as_deref() != Ok("1") {
        eprintln!("skipping attn prefill: set BERTH_GPU_TESTS=1");
        return;
    }
    if !Path::new(LAYERS).is_dir() {
        eprintln!("skipping attn prefill: {LAYERS} is missing");
        return;
    }
    if let Some(why) = port_blocked() {
        eprintln!("skipping attn prefill: {why}");
        return;
    }
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => {
            eprintln!("skipping attn prefill: {why}");
            return;
        }
    };
    // Same switches the fork's bit-exact GEMM path reads. QC staging is
    // chosen inside the prefill wrapper (`q_len >= 256`); the variable is
    // set so a later reader sees the capture condition.
    std::env::set_var("EXL3_HGEMM_F16ACC", "auto");
    std::env::set_var("EXL3_INT8_GEMV", "2");
    std::env::set_var("EXL3_INT8_GEMV_MAX_K", "5");
    std::env::set_var("EXL3_QC_STAGING", "1");

    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let gpu = Gpu::new().unwrap();
    let cc = gpu.ctx.compute_capability().unwrap();
    if gpu.sm_count != 82 || (cc.major, cc.minor) != (8, 6) {
        eprintln!(
            "skipping attn prefill: gemm cache is for sm_86 with 82 SMs (got sm_{}{}, {} SMs); refusing to retune",
            cc.major, cc.minor, gpu.sm_count
        );
        return;
    }

    let cubins = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/triton-cubins");
    let hidden = shape.hidden_size;
    let mut layer = AttnLayer::load(&gpu, &ckpt, &shape, 3, MAX_CHUNK_SIZE).unwrap();
    let mut scratch =
        AttnPrefillScratch::new(&gpu, &shape, MAX_CHUNK_SIZE, MAX_CHUNK_SIZE, &cubins).unwrap();
    let q_out = shape.num_q_heads * shape.head_dim * 2;
    let o_in = shape.num_q_heads * shape.head_dim;
    let mut rows =
        RowScratch::new(&gpu, MAX_CHUNK_SIZE, o_in.max(hidden), q_out.max(hidden)).unwrap();

    layer.reset_cache(&gpu).unwrap();
    compare(
        &gpu,
        &mut layer,
        &mut rows,
        &mut scratch,
        hidden,
        Case {
            name: "100_c0_b03",
            start: 0,
            rows: 99,
        },
    );

    layer.reset_cache(&gpu).unwrap();
    compare(
        &gpu,
        &mut layer,
        &mut rows,
        &mut scratch,
        hidden,
        Case {
            name: "1500_c0_b03",
            start: 0,
            rows: 1280,
        },
    );
    compare(
        &gpu,
        &mut layer,
        &mut rows,
        &mut scratch,
        hidden,
        Case {
            name: "1500_c1_b03",
            start: 1280,
            rows: 219,
        },
    );
}

struct Case {
    name: &'static str,
    start: u32,
    rows: u32,
}

fn compare(
    gpu: &Gpu,
    layer: &mut AttnLayer,
    rows_scratch: &mut RowScratch,
    scratch: &mut AttnPrefillScratch,
    hidden: u32,
    case: Case,
) {
    let Case { name, start, rows } = case;
    let input_bytes = load_bin(&format!("{name}_mixer_in.bin"));
    let want = load_bin(&format!("{name}_mixer.bin"));
    let hidden_u = hidden as usize;
    let rows_u = rows as usize;
    assert_eq!(
        input_bytes.len(),
        rows_u * hidden_u * 2,
        "{name} mixer_in bytes"
    );
    assert_eq!(want.len(), rows_u * hidden_u * 4, "{name} mixer bytes");
    let input = common::upload(gpu, &input_bytes);
    let mut output = gpu.zeroed(want.len()).unwrap();
    layer
        .prefill(gpu, rows_scratch, scratch, &input, &mut output, start, rows)
        .unwrap_or_else(|err| panic!("{name} start {start} rows {rows}: {err}"));
    let got = common::download(gpu, &output);
    let diff = common::differing(&got, &want, 4);
    eprintln!("{name}: {diff} of {} elements differ", want.len() / 4);
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
    assert_eq!(diff, 0, "{name}");
}

fn load_bin(name: &str) -> Vec<u8> {
    let path = Path::new(LAYERS).join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
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
