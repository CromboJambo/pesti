//! Per-layer CPU-vs-dispatch divergence probe.
//!
//! Runs the SAME input through each transformer layer on BOTH the pure-CPU
//! path (transformer::TransformerLayer::forward, all f32) and the dispatch
//! path (kernel::dispatch::LayerDispatch, f16 weights + f16 KV cache), and
//! prints the max |diff| after each layer.
//!
//! Interpretation:
//!   - If max|diff| JUMPS at one specific layer -> structural bug in that
//!     layer's dispatch (attention/FFN/norm).
//!   - If max|diff| grows SLOWLY and monotonically -> f16 precision drift
//!     (expected; the dispatch path stores weights/activations in f16).
//!
//! Usage: cargo run -p pesti-runner --release --features cuda --example probe_layer_diff -- <path>
use std::path::Path;

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_val(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: probe_layer_diff <model.gguf>");
        std::process::exit(2);
    }
    let model_path = args[1].clone();

    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    let model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;
    println!(
        "[config] embed_dim={} vocab={} layers={} heads={} kv_heads={} head_dim={}",
        model.config.embed_dim,
        model.vocab_size,
        model.config.num_layers,
        model.config.num_heads,
        model.config.num_kv_heads,
        model.config.head_dim
    );

    let tok: u32 = 785; // "The"
    let emb = model.embed(tok, 0)?;

    // CPU path: run layer by layer, capturing the hidden state after each.
    let mut cpu_h = emb.clone();
    let mut cpu_after: Vec<Vec<f32>> = Vec::new();
    for layer in model.layers.iter() {
        cpu_h = layer.forward(&cpu_h, 1, 1, 0);
        cpu_after.push(cpu_h.clone());
    }

    // Dispatch path: run layer by layer with the same start_pos=0, capturing
    // the hidden state after each. Rebuilds the LayerDispatch per layer (same
    // as forward_with_dispatch does).
    let ctx = model
        .dispatch
        .as_ref()
        .expect("dispatch context not initialized");

    let mut key_caches = Vec::new();
    let mut value_caches = Vec::new();
    for _ in 0..model.config.num_layers {
        key_caches.push(pesti_runner::kernel::kvcache::Kvcache::new(
            model.config.num_heads,
            model.config.num_kv_heads,
            model.config.head_dim,
            model.config.max_seq_len + 1,
            false,
        ));
        value_caches.push(pesti_runner::kernel::kvcache::Kvcache::new(
            model.config.num_heads,
            model.config.num_kv_heads,
            model.config.head_dim,
            model.config.max_seq_len + 1,
            false,
        ));
    }

    let f32_to_f16 =
        |w: &[f32]| -> Vec<half::f16> { w.iter().map(|&v| half::f16::from_f32(v)).collect() };

    let mut disp_h = emb.clone();
    println!(
        "\n{:<6} {:>14} {:>14} {:>14} {:>12}",
        "layer", "max|cpu|", "max|disp|", "max|diff|", "rel_err"
    );
    for (layer_idx, layer) in model.layers.iter().enumerate() {
        let attention_dispatch = pesti_runner::kernel::dispatch::AttentionDispatch {
            wq: pesti_runner::kernel::dispatch::LinearDispatch::new(
                f32_to_f16(&layer.attention.wq.weight),
                layer.attention.wq.weight.clone(),
                layer.attention.wq.bias.clone(),
                layer.attention.wq.in_features,
                layer.attention.wq.out_features,
            ),
            wk: pesti_runner::kernel::dispatch::LinearDispatch::new(
                f32_to_f16(&layer.attention.wk.weight),
                layer.attention.wk.weight.clone(),
                layer.attention.wk.bias.clone(),
                layer.attention.wk.in_features,
                layer.attention.wk.out_features,
            ),
            wv: pesti_runner::kernel::dispatch::LinearDispatch::new(
                f32_to_f16(&layer.attention.wv.weight),
                layer.attention.wv.weight.clone(),
                layer.attention.wv.bias.clone(),
                layer.attention.wv.in_features,
                layer.attention.wv.out_features,
            ),
            wo: pesti_runner::kernel::dispatch::LinearDispatch::new(
                f32_to_f16(&layer.attention.wo.weight),
                layer.attention.wo.weight.clone(),
                layer.attention.wo.bias.clone(),
                layer.attention.wo.in_features,
                layer.attention.wo.out_features,
            ),
            num_heads: layer.attention.num_heads,
            num_kv_heads: layer.attention.num_kv_heads,
            head_dim: layer.attention.head_dim,
            kv_dim: layer.attention.kv_dim,
            rope_base: layer.attention.rope.base,
        };
        let feed_forward_dispatch = pesti_runner::kernel::dispatch::FeedForwardDispatch {
            w1: pesti_runner::kernel::dispatch::LinearDispatch::new(
                f32_to_f16(&layer.feed_forward.w1.weight),
                layer.feed_forward.w1.weight.clone(),
                layer.feed_forward.w1.bias.clone(),
                layer.feed_forward.w1.in_features,
                layer.feed_forward.w1.out_features,
            ),
            w2: pesti_runner::kernel::dispatch::LinearDispatch::new(
                f32_to_f16(&layer.feed_forward.w2.weight),
                layer.feed_forward.w2.weight.clone(),
                layer.feed_forward.w2.bias.clone(),
                layer.feed_forward.w2.in_features,
                layer.feed_forward.w2.out_features,
            ),
            w3: pesti_runner::kernel::dispatch::LinearDispatch::new(
                f32_to_f16(&layer.feed_forward.w3.weight),
                layer.feed_forward.w3.weight.clone(),
                layer.feed_forward.w3.bias.clone(),
                layer.feed_forward.w3.in_features,
                layer.feed_forward.w3.out_features,
            ),
            intermediate_dim: layer.feed_forward.intermediate_dim,
        };
        let attention_norm = pesti_runner::kernel::dispatch::RmsNormDispatch::new(
            layer.attention_norm.weight.clone(),
            layer.attention_norm.eps,
        );
        let ffn_norm = pesti_runner::kernel::dispatch::RmsNormDispatch::new(
            layer.ffn_norm.weight.clone(),
            layer.ffn_norm.eps,
        );
        let mut layer_dispatch = pesti_runner::kernel::dispatch::LayerDispatch {
            attention: attention_dispatch,
            feed_forward: feed_forward_dispatch,
            attention_norm,
            ffn_norm,
        };

        disp_h = layer_dispatch.forward(
            ctx,
            &disp_h,
            1,
            1,
            0,
            &mut key_caches[layer_idx],
            &mut value_caches[layer_idx],
        )?;

        let cpu = &cpu_after[layer_idx];
        let d = max_abs(cpu, &disp_h);
        let rel = d / max_abs_val(cpu).max(1e-9);
        println!(
            "{:<6} {:>14.6} {:>14.6} {:>14.6} {:>12.4}",
            layer_idx,
            max_abs_val(cpu),
            max_abs_val(&disp_h),
            d,
            rel
        );
    }

    // Final norm + output head comparison (both paths apply final_norm).
    let cpu_final = model
        .final_norm
        .as_ref()
        .map(|n| n.forward(&cpu_h, 1))
        .unwrap_or(cpu_h);
    let disp_final = model
        .final_norm
        .as_ref()
        .map(|n| n.forward(&disp_h, 1))
        .unwrap_or(disp_h);
    let d_final = max_abs(&cpu_final, &disp_final);
    println!(
        "{:<6} {:>14.6} {:>14.6} {:>14.6} {:>12.4}",
        "final",
        max_abs_val(&cpu_final),
        max_abs_val(&disp_final),
        d_final,
        d_final / max_abs_val(&cpu_final).max(1e-9)
    );

    Ok(())
}
