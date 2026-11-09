//! Time the per-step weight conversion vs the actual layer forward.
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let weights = pesti_runner::load_gguf_weights(std::path::Path::new(model_path))?;
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_cfg, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(
        std::path::Path::new(model_path),
        backend,
    )?;
    let prompt_tokens = tokenizer.encode("The quick brown fox.")?;

    // Warm up (KV init) with a short prefill.
    let mut logits: Vec<f32> = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i)?;
        logits = model.forward_with_dispatch(&hidden, i)?;
    }
    let pos = prompt_tokens.len();

    // Time the f32->f16 weight conversion for ONE layer (what
    // forward_with_dispatch does per step, per layer).
    let layer = &model.layers[0];
    let t = Instant::now();
    let _wq = pesti_runner::kernel::dispatch::LinearDispatch::new(
        pesti_runner::transformer::model::f32_to_f16(&layer.attention.wq.weight),
        layer.attention.wq.weight.clone(),
        layer.attention.wq.bias.clone(),
        layer.attention.wq.in_features,
        layer.attention.wq.out_features,
    );
    let wk = pesti_runner::kernel::dispatch::LinearDispatch::new(
        pesti_runner::transformer::model::f32_to_f16(&layer.attention.wk.weight),
        layer.attention.wk.weight.clone(),
        layer.attention.wk.bias.clone(),
        layer.attention.wk.in_features,
        layer.attention.wk.out_features,
    );
    let wv = pesti_runner::kernel::dispatch::LinearDispatch::new(
        pesti_runner::transformer::model::f32_to_f16(&layer.attention.wv.weight),
        layer.attention.wv.weight.clone(),
        layer.attention.wv.bias.clone(),
        layer.attention.wv.in_features,
        layer.attention.wv.out_features,
    );
    let wo = pesti_runner::kernel::dispatch::LinearDispatch::new(
        pesti_runner::transformer::model::f32_to_f16(&layer.attention.wo.weight),
        layer.attention.wo.weight.clone(),
        layer.attention.wo.bias.clone(),
        layer.attention.wo.in_features,
        layer.attention.wo.out_features,
    );
    let w1 = pesti_runner::kernel::dispatch::LinearDispatch::new(
        pesti_runner::transformer::model::f32_to_f16(&layer.feed_forward.w1.weight),
        layer.feed_forward.w1.weight.clone(),
        layer.feed_forward.w1.bias.clone(),
        layer.feed_forward.w1.in_features,
        layer.feed_forward.w1.out_features,
    );
    let w2 = pesti_runner::kernel::dispatch::LinearDispatch::new(
        pesti_runner::transformer::model::f32_to_f16(&layer.feed_forward.w2.weight),
        layer.feed_forward.w2.weight.clone(),
        layer.feed_forward.w2.bias.clone(),
        layer.feed_forward.w2.in_features,
        layer.feed_forward.w2.out_features,
    );
    let w3 = pesti_runner::kernel::dispatch::LinearDispatch::new(
        pesti_runner::transformer::model::f32_to_f16(&layer.feed_forward.w3.weight),
        layer.feed_forward.w3.weight.clone(),
        layer.feed_forward.w3.bias.clone(),
        layer.feed_forward.w3.in_features,
        layer.feed_forward.w3.out_features,
    );
    let conv_ms = t.elapsed().as_secs_f32() * 1000.0;
    println!(
        "[T] layer0 weight f32->f16 + clone (7 linears): {:.2} ms",
        conv_ms
    );
    let _ = (_wq, wk, wv, wo, w1, w2, w3);

    // Time one full forward step for comparison.
    let next = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap() as u32;
    let hidden = model.embed(next, pos)?;
    let t = Instant::now();
    logits = model.forward_with_dispatch(&hidden, pos)?;
    println!("[T] full forward step: {:.3} s", t.elapsed().as_secs_f32());

    // Total weight bytes in f32 across all layers (what gets converted+uploaded
    // every step).
    let total_f32: usize = model
        .layers
        .iter()
        .map(|l| {
            let a = &l.attention;
            let f = &l.feed_forward;
            [&a.wq, &a.wk, &a.wv, &a.wo, &f.w1, &f.w2, &f.w3]
                .iter()
                .map(|lin| lin.weight.len())
                .sum::<usize>()
        })
        .sum::<usize>();
    println!(
        "[T] total layer weight bytes (f32): {total_f32} ({:.2} MB)",
        total_f32 as f64 / 1e6
    );
    println!(
        "[T] if converted+uploaded every step, min time at 50GB/s PCIe: {:.3} s",
        (total_f32 * 2) as f64 / 50e9
    );
    let _ = logits;
    Ok(())
}
