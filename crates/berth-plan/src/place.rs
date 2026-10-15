//! Berth assignment and the per-token byte estimate.
//!
//! Decode speed tracks how many weight bytes a token reads from each berth
//! (`docs/memory.md`). This stage either keeps the whole pack in VRAM or, for
//! a MoE model that does not fit, moves every routed expert to pinned RAM.

use std::collections::HashSet;

use crate::plan::{Ffn, ModelPlan};

/// RTX 3090 framebuffer. 24 GiB.
pub const THIS_MACHINE_VRAM_BYTES: u64 = 24 * 1024 * 1024 * 1024;

/// Pinnable RAM, low end of the 18–20 GB range in `docs/memory.md`. 18 GiB.
pub const THIS_MACHINE_PINNED_RAM_BYTES: u64 = 18 * 1024 * 1024 * 1024;

/// Where one tensor lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Berth {
    /// Device memory.
    Vram,
    /// Host memory pinned for GPU DMA.
    PinnedRam,
    /// Flash-Next only: `ngram_embedding.safetensors` by direct I/O, and cold
    /// expert trellis that did not fit in pinned RAM (`docs/memory.md`).
    ///
    /// [`place`] does not assign this. Stage 4 owns the row format and the
    /// admission policy. Overflow that does not fit in VRAM or pinned RAM is
    /// an error, not an SSD assignment.
    Ssd,
}

/// Caps for [`place`].
///
/// [`Default`] is this machine with nothing reserved. A decode placement
/// should use [`MachineBudget::this_machine`] so the KV pool, GDN state, and
/// workspace are subtracted before weights are packed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MachineBudget {
    pub vram_bytes: u64,
    pub pinned_ram_bytes: u64,
    /// Bytes already reserved in VRAM: KV pool, GDN state, and workspace.
    pub vram_reserved_bytes: u64,
}

impl MachineBudget {
    /// This machine's caps and a caller-supplied VRAM reservation.
    pub fn this_machine(vram_reserved_bytes: u64) -> Self {
        Self {
            vram_bytes: THIS_MACHINE_VRAM_BYTES,
            pinned_ram_bytes: THIS_MACHINE_PINNED_RAM_BYTES,
            vram_reserved_bytes,
        }
    }
}

impl Default for MachineBudget {
    fn default() -> Self {
        Self::this_machine(0)
    }
}

/// Tensor names and byte sizes from the checkpoint.
///
/// Each entry is one tensor the caller wants placed. The size is whatever the
/// reader summed for that name (trellis, or trellis plus `suh` / `svh`).
/// Vision and MTP are not on the decode path; leave them out of this list if
/// they should not take a berth or count toward the per-token estimate.
/// Anything that is listed and is not a routed expert counts as backbone.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TensorBytes {
    pub tensors: Vec<(String, u64)>,
}

impl From<Vec<(String, u64)>> for TensorBytes {
    fn from(tensors: Vec<(String, u64)>) -> Self {
        Self { tensors }
    }
}

/// One placed tensor. Order matches [`TensorBytes::tensors`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacedTensor {
    pub name: String,
    pub bytes: u64,
    pub berth: Berth,
}

/// Byte total for each berth.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BerthBytes {
    pub vram: u64,
    pub pinned_ram: u64,
    pub ssd: u64,
}

impl BerthBytes {
    fn add(&mut self, berth: Berth, bytes: u64) -> Result<(), PlaceError> {
        let slot = match berth {
            Berth::Vram => &mut self.vram,
            Berth::PinnedRam => &mut self.pinned_ram,
            Berth::Ssd => &mut self.ssd,
        };
        *slot = slot.checked_add(bytes).ok_or(PlaceError::Overflow)?;
        Ok(())
    }
}

/// Result of [`place`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub tensors: Vec<PlacedTensor>,
    /// Every listed tensor, summed by the berth it sits in.
    pub resident: BerthBytes,
    /// Bytes a decode token reads from each berth.
    ///
    /// Dense: every listed tensor, from the berth it sits in.
    /// MoE: the backbone, plus `experts_per_tok / num_experts` of the routed
    /// expert bytes. Integer division truncates. Expert bytes that this token
    /// does not read stay in [`Self::resident`] and are not added here.
    pub bytes_moved: BerthBytes,
}

/// [`place`] could not assign berths.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlaceError {
    #[error("tensor name is empty")]
    EmptyName,
    #[error("duplicate tensor name `{0}`")]
    DuplicateTensor(String),
    #[error("tensor byte sizes overflow u64")]
    Overflow,
    #[error("VRAM reservation of {reserved} bytes exceeds the VRAM budget of {vram} bytes")]
    ReservationExceedsVram { reserved: u64, vram: u64 },
    #[error(
        "dense weights ({weights} bytes) plus the VRAM reservation ({reserved} bytes) exceed VRAM ({vram} bytes)"
    )]
    DenseExceedsVram {
        weights: u64,
        reserved: u64,
        vram: u64,
    },
    #[error(
        "backbone ({backbone} bytes) plus the VRAM reservation ({reserved} bytes) exceed VRAM ({vram} bytes)"
    )]
    BackboneExceedsVram {
        backbone: u64,
        reserved: u64,
        vram: u64,
    },
    /// Pinned RAM cannot hold the routed experts. SSD is not a fallback yet.
    #[error(
        "routed experts ({experts} bytes) exceed pinned RAM ({pinned} bytes); SSD placement is not implemented"
    )]
    ExpertsExceedPinnedRam { experts: u64, pinned: u64 },
    #[error("MoE layers disagree on the expert pool")]
    InconsistentMoe,
    #[error("invalid MoE routing: experts_per_tok={experts_per_tok} num_experts={num_experts}")]
    InvalidMoe {
        experts_per_tok: u32,
        num_experts: u32,
    },
}

struct Classed {
    name: String,
    bytes: u64,
    expert: bool,
}

/// True when `name` is a routed-expert tensor.
///
/// A path segment equal to `experts` marks the pool (`mlp.experts.gate_up_proj`,
/// `mlp.experts.0.up_proj`). `shared_expert` and `shared_expert_gate` are backbone.
pub fn is_routed_expert(name: &str) -> bool {
    name.split('.').any(|segment| segment == "experts")
}

/// Place every tensor in `bytes` and estimate per-token traffic.
///
/// The whole pack stays in [`Berth::Vram`] when it fits beside
/// `budget.vram_reserved_bytes`. When it does not, a MoE plan moves every
/// routed expert to [`Berth::PinnedRam`] and leaves the backbone in VRAM. A
/// dense plan returns [`PlaceError::DenseExceedsVram`].
///
/// The hot-expert cache in `docs/memory.md` has no decided size, so this does
/// not keep a subset of experts in VRAM after a spill. [`Berth::Ssd`] is not
/// used; see that variant.
pub fn place(
    plan: &ModelPlan,
    bytes: &TensorBytes,
    budget: &MachineBudget,
) -> Result<Placement, PlaceError> {
    if budget.vram_reserved_bytes > budget.vram_bytes {
        return Err(PlaceError::ReservationExceedsVram {
            reserved: budget.vram_reserved_bytes,
            vram: budget.vram_bytes,
        });
    }
    let available = budget.vram_bytes - budget.vram_reserved_bytes;
    let classed = classify(bytes)?;
    let ratio = moe_ratio(plan)?;
    let total = sum_bytes(&classed, |_| true)?;
    if total <= available {
        return pack(classed, |_| Berth::Vram, ratio);
    }
    let Some(_) = ratio else {
        return Err(PlaceError::DenseExceedsVram {
            weights: total,
            reserved: budget.vram_reserved_bytes,
            vram: budget.vram_bytes,
        });
    };
    // Spill the whole expert pool. A partial hot cache is undecided, and SSD
    // is stage 4, so experts that do not fit in pinned RAM are an error.
    let backbone = sum_bytes(&classed, |tensor| !tensor.expert)?;
    let experts = sum_bytes(&classed, |tensor| tensor.expert)?;
    if backbone > available {
        return Err(PlaceError::BackboneExceedsVram {
            backbone,
            reserved: budget.vram_reserved_bytes,
            vram: budget.vram_bytes,
        });
    }
    if experts > budget.pinned_ram_bytes {
        return Err(PlaceError::ExpertsExceedPinnedRam {
            experts,
            pinned: budget.pinned_ram_bytes,
        });
    }
    pack(classed, spill_berth, ratio)
}

fn spill_berth(tensor: &Classed) -> Berth {
    if tensor.expert {
        Berth::PinnedRam
    } else {
        Berth::Vram
    }
}

fn classify(bytes: &TensorBytes) -> Result<Vec<Classed>, PlaceError> {
    let mut seen = HashSet::with_capacity(bytes.tensors.len());
    let mut out = Vec::with_capacity(bytes.tensors.len());
    for (name, size) in &bytes.tensors {
        if name.is_empty() {
            return Err(PlaceError::EmptyName);
        }
        if !seen.insert(name.as_str()) {
            return Err(PlaceError::DuplicateTensor(name.clone()));
        }
        out.push(Classed {
            name: name.clone(),
            bytes: *size,
            expert: is_routed_expert(name),
        });
    }
    Ok(out)
}

fn moe_ratio(plan: &ModelPlan) -> Result<Option<(u32, u32)>, PlaceError> {
    let mut found = None;
    for layer in &plan.layers {
        let Ffn::Moe {
            router, experts, ..
        } = layer.ffn
        else {
            continue;
        };
        if experts.num_experts == 0
            || experts.experts_per_tok == 0
            || experts.experts_per_tok > experts.num_experts
        {
            return Err(PlaceError::InvalidMoe {
                experts_per_tok: experts.experts_per_tok,
                num_experts: experts.num_experts,
            });
        }
        if router.num_experts != experts.num_experts
            || router.experts_per_tok != experts.experts_per_tok
        {
            return Err(PlaceError::InconsistentMoe);
        }
        let ratio = (experts.experts_per_tok, experts.num_experts);
        match found {
            None => found = Some(ratio),
            Some(prev) if prev != ratio => return Err(PlaceError::InconsistentMoe),
            Some(_) => {}
        }
    }
    Ok(found)
}

fn sum_bytes(classed: &[Classed], keep: impl Fn(&Classed) -> bool) -> Result<u64, PlaceError> {
    let mut total = 0u64;
    for tensor in classed {
        if keep(tensor) {
            total = total
                .checked_add(tensor.bytes)
                .ok_or(PlaceError::Overflow)?;
        }
    }
    Ok(total)
}

fn pack(
    classed: Vec<Classed>,
    berth_of: impl Fn(&Classed) -> Berth,
    ratio: Option<(u32, u32)>,
) -> Result<Placement, PlaceError> {
    let resident = tally(&classed, &berth_of, |_| true)?;
    let bytes_moved = moved_bytes(&classed, &berth_of, ratio)?;
    let tensors = classed
        .into_iter()
        .map(|tensor| {
            let berth = berth_of(&tensor);
            PlacedTensor {
                name: tensor.name,
                bytes: tensor.bytes,
                berth,
            }
        })
        .collect();
    Ok(Placement {
        tensors,
        resident,
        bytes_moved,
    })
}

fn tally(
    classed: &[Classed],
    berth_of: &impl Fn(&Classed) -> Berth,
    keep: impl Fn(&Classed) -> bool,
) -> Result<BerthBytes, PlaceError> {
    let mut out = BerthBytes::default();
    for tensor in classed {
        if keep(tensor) {
            out.add(berth_of(tensor), tensor.bytes)?;
        }
    }
    Ok(out)
}

fn moved_bytes(
    classed: &[Classed],
    berth_of: &impl Fn(&Classed) -> Berth,
    ratio: Option<(u32, u32)>,
) -> Result<BerthBytes, PlaceError> {
    // Dense rereads every listed tensor. MoE rereads the backbone and a
    // top-k slice of the expert pool; the slice is summed per berth, then
    // scaled once so per-tensor truncation does not under-count.
    let Some((per_tok, num_experts)) = ratio else {
        return tally(classed, berth_of, |_| true);
    };
    let mut backbone = tally(classed, berth_of, |tensor| !tensor.expert)?;
    let experts = tally(classed, berth_of, |tensor| tensor.expert)?;
    backbone.add(Berth::Vram, portion(experts.vram, per_tok, num_experts)?)?;
    backbone.add(
        Berth::PinnedRam,
        portion(experts.pinned_ram, per_tok, num_experts)?,
    )?;
    backbone.add(Berth::Ssd, portion(experts.ssd, per_tok, num_experts)?)?;
    Ok(backbone)
}

fn portion(bytes: u64, per_tok: u32, num_experts: u32) -> Result<u64, PlaceError> {
    if num_experts == 0 || per_tok > num_experts {
        return Err(PlaceError::InvalidMoe {
            experts_per_tok: per_tok,
            num_experts,
        });
    }
    let moved = (u128::from(bytes) * u128::from(per_tok)) / u128::from(num_experts);
    u64::try_from(moved).map_err(|_| PlaceError::Overflow)
}
