//! Qwen3.8-27B end to end: checkpoint -> shape -> plan -> placement. Headers only.

use std::path::Path;

use berth_exl3::Checkpoint;
use berth_plan::from_exl3::{model_shape, tensor_bytes};
use berth_plan::{build_plan, place, MachineBudget};

fn model_dir() -> String {
    std::env::var("BERTH_27B_DIR")
        .unwrap_or_else(|_| "/home/sam/qwen38-exl3-dflash2/models/qwen38-27b-exl3".to_string())
}

#[test]
fn qwen38_27b_checkpoint_to_placement() {
    let dir = model_dir();
    if !Path::new(&dir).is_dir() {
        eprintln!("skipping qwen38_27b end to end: directory missing: {dir}");
        return;
    }

    let ckpt = Checkpoint::open(&dir).unwrap();
    let shape = model_shape(&ckpt.config).unwrap();
    let plan = build_plan(&shape).unwrap();
    let bytes = tensor_bytes(&ckpt);
    let placed = place(&plan, &bytes, &MachineBudget::this_machine(0)).unwrap();

    // Text decode path only: 3080 tensors minus 987 vision and 39 MTP, as
    // summed from the headers by a stdlib Python reader.
    assert_eq!(plan.layers.len(), 64);
    assert_eq!(placed.resident.vram, 15_726_411_332);
    assert_eq!(placed.resident.pinned_ram, 0);
    assert_eq!(placed.bytes_moved.vram, 15_726_411_332);
}
