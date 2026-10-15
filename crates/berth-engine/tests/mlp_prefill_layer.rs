//! `MlpLayer::prefill` against `/home/sam/berth/target/prefill_layers`. GPU only.
//!
//! Layers 0 and 3. For each of `100_c0` (99 rows, cooperative GEMM),
//! `1500_c0` (1280 rows, fused reconstruct), and `1500_c1` (219 rows,
//! reconstruct): fp16 `*_bNN_mlp_in` in, fp32 `*_bNN_mlp` out. All 0 differing.

mod common;

use std::path::Path;
use std::process::Command;

use berth_engine::device::RowScratch;
use berth_engine::mlp::{MlpLayer, MlpPrefillScratch};
use berth_engine::Gpu;
use berth_exl3::Checkpoint;

/// Fork dump of blocks 0 and 3. Not in git. See `scripts/dump_27b_prefill_layers.py`.
const PREFILL_DIR: &str = "/home/sam/berth/target/prefill_layers";

/// 27B dense MLP. `meta.json` shapes are `[1, rows, 5120]`.
const HIDDEN: u32 = 5120;
const INTERMEDIATE: u32 = 17408;
/// The model passes 2048 (`MAX_CHUNK_SIZE`).
const MAX_ROWS: u32 = 2048;

/// `(stem, rows)` for the three profiled chunks.
const CASES: &[(&str, u32)] = &[("100_c0", 99), ("1500_c0", 1280), ("1500_c1", 219)];

/// `:8889` is the serve. Same probe as the task: `ss -ltn | grep -q ':8889 '`.
fn serve_blocks_gpu() -> bool {
    let output = match Command::new("ss").args(["-ltn"]).output() {
        Ok(output) => output,
        Err(_) => return true,
    };
    if !output.status.success() {
        return true;
    }
    String::from_utf8_lossy(&output.stdout).contains(":8889 ")
}

fn tensor(name: &str) -> Vec<u8> {
    let path = Path::new(PREFILL_DIR).join(name);
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

#[test]
fn mlp_prefill_matches_the_reference() {
    if !Path::new(PREFILL_DIR).is_dir() {
        eprintln!("skipping: {PREFILL_DIR} is missing");
        return;
    }
    let ckpt_dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => {
            eprintln!("skipping: {why}");
            return;
        }
    };
    if serve_blocks_gpu() {
        eprintln!("skipping: port 8889 is in use or ss failed");
        return;
    }

    // The dump (`dump_27b_prefill_reference.py`) runs under these. `auto`
    // selects the fp16-accumulator GEMM the 1500-token profile measured.
    std::env::set_var("EXL3_INT8_GEMV", "2");
    std::env::set_var("EXL3_INT8_GEMV_MAX_K", "5");
    std::env::set_var("EXL3_QC_STAGING", "1");
    std::env::set_var("EXL3_HGEMM_F16ACC", "auto");

    let ckpt = Checkpoint::open(&ckpt_dir).unwrap();
    let gpu = Gpu::new().unwrap();
    // Down's K and gate's N are both `intermediate`.
    let mut rows_scratch = RowScratch::new(&gpu, MAX_ROWS, INTERMEDIATE, INTERMEDIATE).unwrap();
    let mut scratch = MlpPrefillScratch::new(&gpu, HIDDEN, INTERMEDIATE, MAX_ROWS).unwrap();

    for layer in [0_u32, 3] {
        let mut mlp = MlpLayer::load(&gpu, &ckpt, layer).unwrap();
        for &(stem, rows) in CASES {
            let name = format!("{stem}_b{layer:02}");
            let input = tensor(&format!("{name}_mlp_in.bin"));
            let want = tensor(&format!("{name}_mlp.bin"));
            assert_eq!(
                input.len(),
                rows as usize * HIDDEN as usize * 2,
                "{name}_mlp_in"
            );
            assert_eq!(
                want.len(),
                rows as usize * HIDDEN as usize * 4,
                "{name}_mlp"
            );
            let x = common::upload(&gpu, &input);
            let mut y = gpu.zeroed(want.len()).unwrap();
            mlp.prefill(&gpu, &mut rows_scratch, &mut scratch, &x, &mut y, rows)
                .unwrap();
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
                "{name}_mlp: {n} of {} elements differ; down {}; silu {}",
                want.len() / 4,
                berth_kernels::prefill_lin::last_launch(),
                berth_kernels::ops::last_launch(),
            );
            assert_eq!(n, 0, "{name}_mlp");
            drop(x);
            drop(y);
        }
        // Drop this layer's weights before loading the next one.
        drop(mlp);
    }
}
