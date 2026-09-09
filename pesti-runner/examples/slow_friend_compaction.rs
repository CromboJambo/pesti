//! Slow-Friend G6: Compaction / Re-anchor Probe
//!
//! Proves that periodic re-anchoring of the slow-friend EMA summary keeps
//! long-context drift bounded compared to never compacting.
//!
//! Usage: cargo run -p pesti-runner --release --features cuda \
//!   --example slow_friend_compaction -- <model.gguf> [seq_len]

use std::path::Path;
use std::time::Instant;

use pesti_runner::kernel::slow_friend::{
    divergence, DivergenceMetric, SlowFriendConfig, SlowFriendState, CompactionTrigger, reanchor,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: slow_friend_compaction <model.gguf> [seq_len]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let seq_len: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1024);

    eprintln!("model: {}", model_path);
    eprintln!("seq_len: {}", seq_len);

    // Load model and tokenizer (same pattern as G1 drift probe)
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_, tokenizer) = pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(model_path), backend)?;

    // Build deterministic long prompt by repeating seed
    let seed = "The quick brown fox jumps over the lazy dog. ";
    let mut prompt_tokens = tokenizer.encode(seed)?;
    while prompt_tokens.len() < seq_len {
        let more = tokenizer.encode(seed)?;
        prompt_tokens.extend(more);
    }
    prompt_tokens.truncate(seq_len);

    let hidden_dim = model.config.embed_dim;
    let sf_cfg = SlowFriendConfig { dim: hidden_dim, alpha: 0.95 };

    // === Baseline: No compaction ===
    eprintln!("\n[baseline] Running without compaction...");
    model.reset_cpu_kv_caches();
    let mut slow_base = SlowFriendState::new(&sf_cfg);
    let mut baseline_scores: Vec<f32> = Vec::new();
    let t0 = Instant::now();

    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i)?;
        let h = model.forward_layers_with_cache(&hidden, i)?;
        slow_base.update(&h);
        let score = divergence(DivergenceMetric::Cosine, slow_base.summary(), &h);
        baseline_scores.push(score.value);
    }
    let baseline_elapsed = t0.elapsed();

    // === Compacted: Re-anchor on threshold ===
    eprintln!("[compacted] Running with compaction (threshold=0.1)...");
    model.reset_cpu_kv_caches();
    let mut slow_comp = SlowFriendState::new(&sf_cfg);
    let trigger = CompactionTrigger::new(0.1);
    let mut compacted_scores: Vec<f32> = Vec::new();
    let mut reanchor_count = 0;
    let t1 = Instant::now();

    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i)?;
        let h = model.forward_layers_with_cache(&hidden, i)?;

        // Check divergence before updating slow friend
        if i > 0 {
            let score = divergence(DivergenceMetric::Cosine, slow_comp.summary(), &h);
            compacted_scores.push(score.value);
            if trigger.should_compact(&score) {
                reanchor(&mut slow_comp, &h, 1.0); // full reset to precise state
                reanchor_count += 1;
            } else {
                slow_comp.update(&h);
            }
        } else {
            slow_comp.update(&h);
            compacted_scores.push(0.0);
        }
    }
    let compacted_elapsed = t1.elapsed();

    // === Results ===
    println!("\nSlow-Friend G6: Compaction Probe");
    println!("=================================");
    println!("model: {}", model_path);
    println!("seq_len: {}", seq_len);
    println!("hidden_dim: {}", hidden_dim);
    println!();

    let baseline_mean = baseline_scores.iter().sum::<f32>() / baseline_scores.len() as f32;
    let baseline_max = baseline_scores.iter().cloned().fold(f32::MIN, |a, b| a.max(b));
    let baseline_tail = tail_mean(&baseline_scores);

    let comp_mean = compacted_scores.iter().sum::<f32>() / compacted_scores.len() as f32;
    let comp_max = compacted_scores.iter().cloned().fold(f32::MIN, |a, b| a.max(b));
    let comp_tail = tail_mean(&compacted_scores);

    println!("Condition     | mean_div   | max_div    | tail_mean  | reanchors");
    println!("--------------|------------|------------|------------|----------");
    println!("baseline      | {:>10.6} | {:>10.6} | {:>10.6} | {}", baseline_mean, baseline_max, baseline_tail, 0);
    println!("compacted     | {:>10.6} | {:>10.6} | {:>10.6} | {}", comp_mean, comp_max, comp_tail, reanchor_count);

    let improvement = if baseline_mean > 0.0 {
        (baseline_mean - comp_mean) / baseline_mean * 100.0
    } else {
        0.0
    };
    println!();
    println!("Mean divergence reduction: {:.2}%", improvement);

    // G6 Verdict
    println!("\nG6 Verdict:");
    if comp_tail < baseline_tail {
        println!("PASS(G6): Compaction reduces final drift ({:.6} -> {:.6})", baseline_tail, comp_tail);
    } else {
        println!("FAIL(G6): Compaction did not reduce final drift ({:.6} vs {:.6})", baseline_tail, comp_tail);
    }

    eprintln!("\nBaseline elapsed: {:.3}s", baseline_elapsed.as_secs_f64());
    eprintln!("Compacted elapsed: {:.3}s", compacted_elapsed.as_secs_f64());

    Ok(())
}

fn tail_mean(scores: &[f32]) -> f32 {
    let tail_start = (scores.len() as f64 * 0.9).ceil() as usize;
    let tail = &scores[tail_start..];
    if tail.is_empty() {
        scores.iter().sum::<f32>() / scores.len() as f32
    } else {
        tail.iter().sum::<f32>() / tail.len() as f32
    }
}