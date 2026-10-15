//! Embedding, norms, residual glue, and `lm_head` against `testdata/decode_ref`.
//!
//! No mixers and no layer stack. The weights are the reference token's
//! embedding rows, the layer-0 and final RMSNorms, and `lm_head`.
//! GPU only: `BERTH_GPU_TESTS=1`, and skipped when port 8889 is listening.

mod common;

use std::io::{Read, Seek, SeekFrom};
use std::process::Command;

use berth_engine::model::{Model, TextNorm};
use berth_engine::{upload_tensor, Gpu};
use berth_exl3::{Checkpoint, DType, TensorInfo};

#[test]
fn model_parts_match_decode_ref() {
    let dir = match common::gpu_ready() {
        Ok(dir) => dir,
        Err(why) => return eprintln!("skipping: {why}"),
    };
    if serve_listening() {
        return eprintln!("skipping: port 8889 is in use");
    }

    let ckpt = Checkpoint::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();
    let norm = TextNorm::qwen3_5(1e-6);
    let token = common::reference_token();

    let want_embed = common::reference("embedding");
    let hidden = i32::try_from(want_embed.len() / 4).unwrap();
    assert_eq!(want_embed.len(), hidden as usize * 4);
    let got_embed = embed_token(&gpu, &ckpt, token, hidden);
    report("embedding", &got_embed, &want_embed, 4);

    let want_mixer_in = common::reference("block_00_mixer_in");
    let mixer_in = norm_of(
        &gpu,
        &ckpt,
        "model.language_model.layers.0.input_layernorm.weight",
        &want_embed,
        hidden,
        norm,
    );
    report("block_00_mixer_in", &mixer_in, &want_mixer_in, 2);

    let want_final = common::reference("final_norm");
    let block_63 = common::reference("block_63");
    let final_norm = norm_of(
        &gpu,
        &ckpt,
        "model.language_model.norm.weight",
        &block_63,
        hidden,
        norm,
    );
    report("final_norm", &final_norm, &want_final, 2);

    glue_block0(&gpu, &ckpt, &want_embed, hidden, norm);

    // `lm_head` is about 1 GB. Run it last so the small buffers are already gone.
    let want_logits = common::reference("logits");
    let mut head = berth_engine::Exl3Linear::load(&gpu, &ckpt, "lm_head").unwrap();
    let x = common::upload(&gpu, &want_final);
    let mut logits = gpu.zeroed(head.size_n as usize * 2).unwrap();
    Model::project_lm_head(&gpu, &mut head, &x, &mut logits).unwrap();
    let got_logits = common::download(&gpu, &logits);
    eprintln!("lm_head kernel: {}", berth_kernels::last_launch());
    report("logits", &got_logits, &want_logits, 2);
}

fn glue_block0(gpu: &Gpu, ckpt: &Checkpoint, embedding: &[u8], hidden: i32, norm: TextNorm) {
    let want_mlp_in = common::reference("block_00_mlp_in");
    let mixer = common::reference("block_00_mixer");
    let want_block = common::reference("block_00");
    let mlp = common::reference("block_00_mlp");

    let weight = upload_tensor(
        gpu,
        ckpt,
        "model.language_model.layers.0.post_attention_layernorm.weight",
    )
    .unwrap();
    let mut residual = common::upload(gpu, embedding);
    let mixer_buf = common::upload(gpu, &mixer);
    let mut mlp_in = gpu.zeroed(hidden as usize * 2).unwrap();
    Model::norm_residual(
        gpu,
        &mixer_buf,
        &weight,
        &mut mlp_in,
        &mut residual,
        hidden,
        norm,
    )
    .unwrap();
    let got_mlp_in = common::download(gpu, &mlp_in);
    report("block_00_mlp_in", &got_mlp_in, &want_mlp_in, 2);

    let got_residual = common::download(gpu, &residual);
    let expect_residual = add_f32_host(embedding, &mixer);
    report(
        "block_00 post-norm residual",
        &got_residual,
        &expect_residual,
        4,
    );

    let mlp_buf = common::upload(gpu, &mlp);
    Model::add_residual_in_place(gpu, &mut residual, &mlp_buf, hidden).unwrap();
    let got_block = common::download(gpu, &residual);
    report("block_00", &got_block, &want_block, 4);
}

fn embed_token(gpu: &Gpu, ckpt: &Checkpoint, token: u32, hidden: i32) -> Vec<u8> {
    let name = "model.language_model.embed_tokens.weight";
    let info = ckpt.tensors.get(name).unwrap_or_else(|| panic!("{name}"));
    assert_eq!(info.dtype, DType::Bf16);
    assert_eq!(info.shape.len(), 2);
    assert_eq!(info.shape[1], hidden as u64);
    assert!(
        u64::from(token) < info.shape[0],
        "token {token} past vocab {}",
        info.shape[0]
    );
    let row = hidden as u64 * 2;
    // Rows `0..=token`, so the kernel indexes the real token id.
    let prefix = read_range(info, 0, (u64::from(token) + 1) * row);
    let weight = common::upload(gpu, &prefix);
    let mut out = gpu.zeroed(hidden as usize * 4).unwrap();
    Model::embed_token(gpu, &weight, token, hidden, &mut out).unwrap();
    eprintln!("embed kernel: {}", berth_kernels::ops::last_launch());
    common::download(gpu, &out)
}

fn norm_of(
    gpu: &Gpu,
    ckpt: &Checkpoint,
    weight_name: &str,
    x: &[u8],
    hidden: i32,
    norm: TextNorm,
) -> Vec<u8> {
    let weight = upload_tensor(gpu, ckpt, weight_name).unwrap();
    let x = common::upload(gpu, x);
    let mut y = gpu.zeroed(hidden as usize * 2).unwrap();
    Model::norm_fp32_to_fp16(gpu, &x, &weight, &mut y, hidden, norm).unwrap();
    eprintln!("{weight_name}: {}", berth_kernels::ops::last_launch());
    common::download(gpu, &y)
}

fn report(name: &str, got: &[u8], want: &[u8], elem: usize) {
    let diff = common::differing(got, want, elem);
    let n = want.len() / elem;
    eprintln!("{name}: {diff} of {n} elements differ");
    assert_eq!(diff, 0, "{name}");
}

fn add_f32_host(a: &[u8], b: &[u8]) -> Vec<u8> {
    assert_eq!(a.len(), b.len());
    a.chunks(4)
        .zip(b.chunks(4))
        .flat_map(|(left, right)| {
            let left = f32::from_le_bytes(left.try_into().unwrap());
            let right = f32::from_le_bytes(right.try_into().unwrap());
            (left + right).to_le_bytes()
        })
        .collect()
}

fn read_range(info: &TensorInfo, start: u64, len: u64) -> Vec<u8> {
    let mut file = std::fs::File::open(&info.shard)
        .unwrap_or_else(|err| panic!("{}: {err}", info.shard.display()));
    file.seek(SeekFrom::Start(info.offset + start))
        .unwrap_or_else(|err| panic!("seek {}: {err}", info.name));
    let mut bytes = vec![0u8; len as usize];
    file.read_exact(&mut bytes)
        .unwrap_or_else(|err| panic!("read {}: {err}", info.name));
    bytes
}

fn serve_listening() -> bool {
    let output = match Command::new("ss").args(["-ltn"]).output() {
        Ok(output) => output,
        Err(_) => return true,
    };
    if !output.status.success() {
        return true;
    }
    String::from_utf8_lossy(&output.stdout).contains(":8889 ")
}
