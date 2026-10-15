//! Forward pass and decode loop.
//!
//! Runs a `LayerPlan`, captures CUDA graphs, samples on the GPU, and hands each
//! accepted token to the repetition check. Draft accept / rewind lands here later.
//! Tests that touch the GPU run only with `BERTH_GPU_TESTS=1`.
//! Spec: `docs/runtime.md`.
//!
//! Stage 1 step 5 is one eager decode step of the Qwen3.8-27B, checked block by
//! block against `testdata/decode_ref/`. Graph capture comes after that.
//!
//! One block, as the fork's `TransformerBlock.forward` runs it at decode:
//!
//! ```text
//! h  = rms_norm(x)                 input_layernorm, fp32 x -> fp16 h
//! y  = mixer(h)                    GdnLayer or AttnLayer, fp16 -> fp32
//! h2 = rms_norm_res_in(x, y)       post_attention_layernorm; x += y in place
//! m  = mlp(h2)                     MlpLayer, fp16 -> fp32
//! x  = x + m                       add_f32
//! ```

pub mod device;
mod error;

pub mod attn;
pub mod gdn;
pub mod mlp;
pub mod model;

pub use device::{upload_tensor, Exl3Linear, Gpu};
pub use error::{Error, Result};
