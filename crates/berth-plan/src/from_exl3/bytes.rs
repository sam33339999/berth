//! Checkpoint tensors → [`crate::TensorBytes`].
//!
//! One entry per quantized linear, plus one entry per tensor that is not part
//! of a linear. Vision and MTP stay out: they are not on the text decode path
//! (`docs/runtime.md`).

use berth_exl3::{Checkpoint, LinearExl3};

use crate::TensorBytes;

/// Vision-tower tensors in the Qwen3.8-27B EXL3 header.
///
/// `python3 crates/berth-exl3/tests/dump_headers.py` on that checkpoint
/// (3080 tensors) prints 987 names under `model.visual.` and no other vision
/// prefix: `model.visual.blocks.*` (972), `model.visual.merger.*` (12),
/// `model.visual.patch_embed.proj.{weight,bias}`, and
/// `model.visual.pos_embed.weight`. `model.vision_tower` does not appear.
pub const VISION_TOWER_PREFIX: &str = "model.visual.";

/// MTP tensors in the Qwen3.8-27B EXL3 header.
///
/// The same dump prints 39 names under `mtp.` and no other MTP prefix:
/// `mtp.fc.{trellis,suh,svh,mul1}`, one `mtp.layers.0` decoder layer,
/// `mtp.norm.weight`, `mtp.pre_fc_norm_embedding.weight`, and
/// `mtp.pre_fc_norm_hidden.weight`.
pub const MTP_PREFIX: &str = "mtp.";

/// Suffixes `group_linears` consumes for one [`LinearExl3`] prefix.
///
/// Same set as `LINEAR_PARTS` in `crates/berth-exl3/src/linear.rs`. For each
/// linear, every `{prefix}.{suffix}` the index contains is removed from
/// [`Checkpoint::other_tensors`], so those bytes belong on this entry only.
const LINEAR_PARTS: &[&str] = &["trellis", "suh", "svh", "su", "sv", "bias", "mcg", "mul1"];

/// Sum checkpoint tensors into the decode-path [`TensorBytes`].
///
/// One entry per quantized linear, named by [`LinearExl3::prefix`]. Its size
/// is the sum of `length` over every tensor that linear owns: `trellis`, the
/// input scale (`suh` or packed `su`), the output scale (`svh` or packed
/// `sv`), optional `bias`, and `mcg` or `mul1` when that marker is its own
/// tensor. [`berth_exl3::HadamardScale`] records only one scale per side
/// (unpacked wins). A checkpoint that stored both still owns both:
/// the packed tensor is not an `other_tensors` entry, so it is added here.
///
/// One entry per remaining [`Checkpoint::other_tensors`] record, under that
/// tensor's own name (norms, token embedding, `conv1d` weights, `A_log`,
/// `dt_bias`, and other unquantized weights).
///
/// Names that start with [`VISION_TOWER_PREFIX`] or [`MTP_PREFIX`] are left
/// out. Routed-expert linears keep the stored prefix, including the
/// `.experts.` segment [`crate::is_routed_expert`] matches. There is no local
/// EXL3 MoE checkpoint; this function does not rename experts.
///
/// Entries are sorted by name. That is not checkpoint order: [`Checkpoint::linears`]
/// and [`Checkpoint::other_tensors`] are two lists. Only `TensorInfo::length`
/// is read; tensor bodies stay on disk.
pub fn tensor_bytes(checkpoint: &Checkpoint) -> TensorBytes {
    let mut tensors = Vec::new();
    for linear in checkpoint.linears() {
        if off_decode_path(&linear.prefix) {
            continue;
        }
        tensors.push((linear.prefix.clone(), linear_bytes(checkpoint, linear)));
    }
    for info in checkpoint.other_tensors() {
        if off_decode_path(&info.name) {
            continue;
        }
        tensors.push((info.name.clone(), info.length));
    }
    tensors.sort_by(|left, right| left.0.cmp(&right.0));
    TensorBytes { tensors }
}

fn off_decode_path(name: &str) -> bool {
    name.starts_with(VISION_TOWER_PREFIX) || name.starts_with(MTP_PREFIX)
}

/// Bytes of every `LINEAR_PARTS` tensor the index still has for this prefix.
///
/// A missing suffix is a form this linear did not store (`su` when the scale
/// is unpacked `suh`, `mcg` when the codebook is `mul1` or the default).
/// Present suffixes are disjoint names, so each tensor is added once.
fn linear_bytes(checkpoint: &Checkpoint, linear: &LinearExl3) -> u64 {
    let mut total = 0u64;
    for suffix in LINEAR_PARTS {
        let name = format!("{}.{suffix}", linear.prefix);
        if let Some(info) = checkpoint.tensors.get(&name) {
            total += info.length;
        }
    }
    total
}
