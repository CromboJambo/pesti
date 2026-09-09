//! Slow-Friend Drift Probe (G1)
//!
//! Measures whether divergence between a stable low-pass reference and the
//! precise path grows with context length. Uses CPU-only inference to avoid
//! GPU OOM issues while still exercising the real forward pass.
//!
//! Usage: cargo run -p pesti-runner --release --features cuda \
//!   --example slow_friend_drift -- <model.gguf> [seq_lengths...]

use std::path::Path;
use std::time::Instant;

use pesti_runner::kernel::slow_friend::{divergence, DivergenceMetric, SlowFriendConfig, SlowFriendState};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: slow_friend_drift <model.gguf> [seq_lengths...]");
        std::process::exit(2);
    }
    let model_path = &args[1];

    // Parse sequence lengths from env or args
    let seq_lens_str = match args.get(2) {
        Some(s) => s.clone(),
        None => std::env::var("PESTI_SLOW_SEQS").unwrap_or_else(|_| "512,1024,2048".into()),
    };
    let seq_lens: Vec<usize> = seq_lens_str.split(',').map(|s| s.trim().parse())
        .collect::<Result<Vec<usize>, _>>()?;

    eprintln!("model: {}", model_path);
    eprintln!("seq_lengths: {:?}", seq_lens);

    // Load model and tokenizer using the same pattern as cpu_e2e_generate.rs
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(model_path), backend)?;

    // Seed sentence for deterministic long prompts
    let seed = "The quick brown fox jumps over the lazy dog. ";

    println!("\nSlow-Friend Drift Probe (G1)");
    println!("============================");
    println!("seq_len | mean_d     | max_d      | tail_mean_d | slow_us/step");
    println!("--------|------------|------------|-------------|-------------");

    let mut results: Vec<(usize, f32, f32, f32)> = Vec::new();

    for seq_len in &seq_lens {
        // Build deterministic long prompt by repeating seed
        let mut prompt_tokens = tokenizer.encode(seed)?;
        while prompt_tokens.len() < *seq_len {
            let more = tokenizer.encode(seed)?;
            prompt_tokens.extend(more);
        }
        prompt_tokens.truncate(*seq_len);

        eprintln!("\n[seq={}] Running forward pass...", seq_len);
        model.reset_cpu_kv_caches();

        let hidden_dim = model.config.embed_dim;
        let sf_cfg = SlowFriendConfig { dim: hidden_dim, alpha: 0.95 };
        let mut slow = SlowFriendState::new(&sf_cfg);

        let mut div_scores: Vec<f32> = Vec::new();
        let t_start = Instant::now();
        let mut sf_total_us = 0;

        for (i, &tok) in prompt_tokens.iter().enumerate() {
            let sf_t0 = Instant::now();

            let hidden = model.embed(tok, i)?;
            let h = model.forward_layers_with_cache(&hidden, i)?;

            // Update slow friend with this step's hidden state
            slow.update(&h);

            // Compute divergence from stable summary to current precise state
            let score = divergence(DivergenceMetric::Cosine, slow.summary(), &h);
            div_scores.push(score.value);

            sf_total_us += sf_t0.elapsed().as_micros() as u64;

            // Only compute logits for last token (we're measuring drift, not generating)
            if i == prompt_tokens.len() - 1 {
                let _logits = model.apply_output_head(&h)?;
            }
        }

        let elapsed = t_start.elapsed();
        let mean_d = div_scores.iter().sum::<f32>() / div_scores.len() as f32;
        let max_d = div_scores.iter().cloned().fold(f32::MIN, |a, b| a.max(b));

        // Tail mean: last 10% of steps
        let tail_start = (div_scores.len() as f64 * 0.9).ceil() as usize;
        let tail_scores = &div_scores[tail_start..];
        let tail_mean_d = if tail_scores.is_empty() {
            mean_d
        } else {
            tail_scores.iter().sum::<f32>() / tail_scores.len() as f32
        };

        let sf_avg_us = sf_total_us as f64 / div_scores.len() as f64;

        println!(
            "   {:>4}  | {:>10.6} | {:>10.6} | {:>11.6} | {:>8.2}",
            seq_len, mean_d, max_d, tail_mean_d, sf_avg_us
        );

        results.push((*seq_len, mean_d, max_d, tail_mean_d));

        eprintln!(
            "[seq={}] done in {:.3}s ({} steps, avg {:.2}us/step for slow-friend ops)",
            seq_len, elapsed.as_secs_f64(), div_scores.len(), sf_avg_us
        );
    }

    // G1 Verdict: is tail_mean_d non-decreasing across seq_len ladder?
    println!("\nG1 Verdict:");
    let mut pass = true;
    for i in 1..results.len() {
        let (prev_seq, _, _, prev_tail) = results[i - 1];
        let (seq, _, _, tail) = results[i];
        if tail < prev_tail * 0.95 {
            // allow 5% tolerance for noise
            pass = false;
            println!(
                "  FAIL: tail_mean_d decreased from seq {} ({:.6}) to seq {} ({:.6})",
                prev_seq, prev_tail, seq, tail
            );
        } else {
            println!(
                "  OK: seq {} -> {} : {:.6} <= {:.6}",
                prev_seq, seq, prev_tail, tail
            );
        }
    }

    if pass {
        println!("PASS(G1): Divergence signal grows with context length");
    } else {
        println!("FAIL(G1): Divergence signal does not grow monotonically");
    }

    Ok(())
}
