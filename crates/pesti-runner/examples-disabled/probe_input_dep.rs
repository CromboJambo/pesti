//! Probe: isolate WHERE input-dependence dies in the forward pass.
//!
//! Embeds two different tokens, runs them through the full forward pass,
//! and prints hidden-state and logit statistics for each. If the hidden
//! states are identical (or NaN), the bug is in the transformer layers.
//! If the hidden states differ but logits are constant, the bug is in the
//! output head.
//!
//! Usage: cargo run -p pesti-runner --release --features cuda --example probe_input_dep -- <path>

use std::path::Path;

fn stats(label: &str, vals: &[f32]) {
    let n = vals.len();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut nan_count = 0;
    let mut sum = 0.0f64;
    for &v in vals {
        if v.is_nan() {
            nan_count += 1;
        } else {
            min = min.min(v);
            max = max.max(v);
            sum += v as f64;
        }
    }
    let mean = if n > nan_count {
        sum / (n - nan_count) as f64
    } else {
        f64::NAN
    };
    println!(
        "  {:<28} n={:<7} min={:+.6} max={:+.6} mean={:+.6} nan={}",
        label, n, min, max, mean, nan_count
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: probe_input_dep <model.gguf>");
        std::process::exit(2);
    }
    let model_path = args[1].clone();

    let t0 = std::time::Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    println!(
        "[load] {:.2}s tensors={}",
        t0.elapsed().as_secs_f32(),
        weights.tensors.len()
    );

    let t1 = std::time::Instant::now();
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;
    println!("[build] {:.2}s", t1.elapsed().as_secs_f32());
    println!(
        "[config] embed_dim={} vocab={} layers={}",
        model.config.embed_dim, model.vocab_size, model.config.num_layers
    );

    // Two very different tokens
    let tok_a: u32 = 785; // "The"
    let tok_b: u32 = 3974; // "quick"

    for (label, tok) in [("token 785 (The)", tok_a), ("token 3974 (quick)", tok_b)] {
        println!("\n=== {} ===", label);

        // 1. Embedding
        let emb = model.embed(tok, 0)?;
        stats(&format!("embedding[{}]", tok), &emb);

        // 2. Run through layers (CPU path, no KV cache, no dispatch)
        //    This isolates the transformer layers from the dispatch/GEMM path.
        let hidden_cpu = model.forward_layers(&emb, 0)?;
        stats("hidden_after_layers(cpu)", &hidden_cpu);

        // 3. Output head on CPU hidden state
        let logits_cpu = model.apply_output_head(&hidden_cpu)?;
        stats("logits_cpu", &logits_cpu);
        let top_cpu = pesti_runner::transformer::argmax(&logits_cpu);
        println!("  argmax(cpu) = {}", top_cpu);

        // 4. Now the dispatch path (what generate() actually uses)
        model.reset_cpu_kv_caches();
        let logits_disp = model.forward_with_dispatch(&emb, 0)?;
        stats("logits_dispatch", &logits_disp);
        let top_disp = pesti_runner::transformer::argmax(&logits_disp);
        println!("  argmax(dispatch) = {}", top_disp);

        // 5. Compare CPU vs dispatch logits
        if logits_cpu.len() == logits_disp.len() {
            let diff: f32 = logits_cpu
                .iter()
                .zip(logits_disp.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("  max|cpu-dispatch| = {:.6}", diff);
        }
    }

    // 6. Cross-check: do the two tokens produce DIFFERENT embeddings?
    let emb_a = model.embed(tok_a, 0)?;
    let emb_b = model.embed(tok_b, 0)?;
    let emb_diff: f32 = emb_a
        .iter()
        .zip(emb_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("\n=== cross-check ===");
    println!("  max|emb_a - emb_b| = {:.6} (should be > 0)", emb_diff);

    // 7. Check if the output head weight is all-zeros or degenerate
    if let Some(ref out) = model.output {
        let w = &out.weight;
        let mut w_min = f32::INFINITY;
        let mut w_max = f32::NEG_INFINITY;
        let mut w_nonzero = 0;
        for &v in w {
            if v != 0.0 {
                w_nonzero += 1;
            }
            w_min = w_min.min(v);
            w_max = w_max.max(v);
        }
        println!(
            "  output.weight: n={} nonzero={} min={:+.6} max={:+.6}",
            w.len(),
            w_nonzero,
            w_min,
            w_max
        );
    }

    Ok(())
}
