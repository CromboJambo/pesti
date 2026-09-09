//! Slow-Friend G7: End-to-End Integration
//!
//! Integrates the slow-friend substrate into the actual decode loop and proves:
//! 1. Non-interference: token sequences are identical with/without slow-friend hook
//! 2. Divergence tracking: scores are computed at each step
//! 3. Perturbation sensitivity: induced drift is detected as higher divergence
//!
//! Usage: ./slow_friend_e2e <model.gguf> [max_tokens]

use std::path::Path;
use std::time::Instant;

use pesti_runner::kernel::slow_friend::{
    divergence, DivergenceMetric, SlowFriendConfig, SlowFriendState,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: slow_friend_e2e <model.gguf> [max_tokens]");
        std::process::exit(2);
    }
    let model_path = &args[1];
    let max_tokens: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);

    println!("=== Slow-Friend G7: End-to-End Integration ===");
    println!();

    // Load model and tokenizer (same pattern as slow_friend_drift.rs)
    println!("Loading model: {}", model_path);
    let load_start = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;
    println!("Model loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_, tokenizer) =
        pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(model_path), backend)?;

    // Use Qwen2.5-Instruct chat template format
    let prompt_text = "< |im_start|>user\nWrite a short poem about mountains.< |im_end|>\n< |im_start|>assistant\n";
    let input_tokens = tokenizer.encode(prompt_text)?;

    println!("Prompt: {} tokens", input_tokens.len());
    println!();

    // === Test 1: Baseline generation (no slow-friend hook) ===
    println!("=== Test 1: Baseline Generation ===");
    model.reset_cpu_kv_caches();
    let gen_start = Instant::now();
    let baseline_tokens = generate_baseline(&mut model, &input_tokens, max_tokens)?;
    let gen_time = gen_start.elapsed().as_secs_f64();
    println!("Baseline generated {} tokens in {:.1}s", baseline_tokens.len(), gen_time);

    // Decode baseline text
    let baseline_text = tokenizer.decode(&baseline_tokens)?;
    println!("Baseline text: {}", &baseline_text[..std::cmp::min(80, baseline_text.len())]);
    println!();

    // === Test 2: Generation with slow-friend hook (observational) ===
    println!("=== Test 2: Slow-Friend Integrated Generation ===");

    let hidden_dim = model.config.embed_dim;
    let sf_cfg = SlowFriendConfig { dim: hidden_dim, alpha: 0.95 };
    let mut sf_state = SlowFriendState::new(&sf_cfg);
    let mut divergence_scores = Vec::new();

    model.reset_cpu_kv_caches();
    let gen_start = Instant::now();
    let sf_tokens = generate_with_sf_hook(
        &mut model,
        &input_tokens,
        max_tokens,
        &mut sf_state,
        &mut divergence_scores,
    )?;
    let gen_time = gen_start.elapsed().as_secs_f64();
    println!("Slow-friend generated {} tokens in {:.1}s", sf_tokens.len(), gen_time);

    // === Non-interference check: tokens must be identical ===
    println!();
    println!("=== Non-Interference Check ===");
    if baseline_tokens == sf_tokens {
        println!("PASS(G7-1): Token sequences are bit-identical ({} tokens)", baseline_tokens.len());
    } else {
        // Find first divergence point
        let mut first_diff = 0;
        for (i, (&b, &s)) in baseline_tokens.iter().zip(sf_tokens.iter()).enumerate() {
            if b != s {
                first_diff = i;
                break;
            }
        }
        println!("FAIL(G7-1): Token sequences differ at position {}", first_diff);
    }

    // === Divergence score analysis ===
    println!();
    println!("=== Divergence Score Analysis ===");
    if !divergence_scores.is_empty() {
        let sum: f32 = divergence_scores.iter().sum();
        let mean = sum / divergence_scores.len() as f32;
        let max = divergence_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = divergence_scores.iter().cloned().fold(f32::INFINITY, f32::min);

        println!("Divergence scores computed: {}", divergence_scores.len());
        println!("  Mean: {:.6}", mean);
        println!("  Min:  {:.6}", min);
        println!("  Max:  {:.6}", max);

        // Show trend: first 10 vs last 10
        let n = std::cmp::min(10, divergence_scores.len());
        if n > 0 {
            let early_sum: f32 = divergence_scores[..n].iter().sum();
            let late_start = divergence_scores.len() - n;
            let late_sum: f32 = divergence_scores[late_start..].iter().sum();
            println!("  Early mean (first {}): {:.6}", n, early_sum / n as f32);
            println!("  Late mean (last {}):   {:.6}", n, late_sum / n as f32);
        }
    } else {
        println!("FAIL(G7-2): No divergence scores computed");
    }

    // === Test 3: Drift induction (perturbation test) ===
    println!();
    println!("=== Test 3: Drift Induction Test ===");

    let mut sf_state2 = SlowFriendState::new(&sf_cfg);
    let mut perturbed_divergences = Vec::new();
    let perturbation_step = 10; // Start perturbing at step 10

    model.reset_cpu_kv_caches();
    generate_with_perturbation(
        &mut model,
        &input_tokens,
        max_tokens,
        &mut sf_state2,
        &mut perturbed_divergences,
        perturbation_step,
    )?;

    // Compare divergence during perturbation window vs baseline
    let baseline_window: Vec<f32> = divergence_scores[perturbation_step..std::cmp::min(perturbation_step + 10, divergence_scores.len())].to_vec();
    let perturbed_window: Vec<f32> = perturbed_divergences.iter()
        .filter(|(p, _)| *p >= perturbation_step && *p < perturbation_step + 10)
        .map(|(_, v)| *v)
        .collect();

    if !baseline_window.is_empty() && !perturbed_window.is_empty() {
        let baseline_mean: f32 = baseline_window.iter().sum::<f32>() / baseline_window.len() as f32;
        let perturbed_mean: f32 = perturbed_window.iter().sum::<f32>() / perturbed_window.len() as f32;

        println!("Baseline divergence (steps {}-{}): {:.6}",
                 perturbation_step, perturbation_step + 10, baseline_mean);
        println!("Perturbed divergence (steps {}-{}): {:.6}",
                 perturbation_step, perturbation_step + 10, perturbed_mean);

        if perturbed_mean > baseline_mean * 2.0 {
            println!("PASS(G7-3): Perturbation detected - {:.1}x higher divergence",
                     perturbed_mean / baseline_mean);
        } else {
            println!("FAIL(G7-3): Perturbation not clearly detected ({:.1}x baseline)",
                     perturbed_mean / baseline_mean);
        }
    } else {
        println!("FAIL(G7-3): Insufficient divergence data for perturbation test");
    }

    // === Final verdict ===
    println!();
    println!("=== G7 Verdict ===");
    let pass_1 = baseline_tokens == sf_tokens;
    let pass_2 = !divergence_scores.is_empty();
    let pass_3 = perturbed_window.iter().sum::<f32>() / perturbed_window.len() as f32 >
                baseline_window.iter().sum::<f32>() / baseline_window.len() as f32 * 2.0;

    if pass_1 && pass_2 && pass_3 {
        println!("PASS(G7): All E2E integration criteria met");
    } else {
        println!("FAIL(G7): Some criteria not met");
        if !pass_1 { println!("  - Non-interference violated"); }
        if !pass_2 { println!("  - No divergence scores computed"); }
        if !pass_3 { println!("  - Perturbation not detected"); }
    }

    Ok(())
}

/// Generate tokens without slow-friend hook (baseline)
fn generate_baseline(
    model: &mut pesti_runner::transformer::LlamaModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut generated = Vec::new();
    let mut last_token = prompt_tokens.last().copied().unwrap_or(0);

    // Forward pass on prompt
    model.reset_cpu_kv_caches();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i)?;
        let h = model.forward_layers_with_cache(&hidden, i)?;
        if i == prompt_tokens.len() - 1 {
            last_token = tok;
        }
    }

    // Generate tokens
    for step in 0..max_tokens {
        let hidden = model.embed(last_token, prompt_tokens.len() + step)?;
        let h = model.forward_layers_with_cache(&hidden, prompt_tokens.len() + step)?;
        let logits = model.apply_output_head(&h)?;

        // Greedy sampling (deterministic)
        let next_token = argmax(&logits);
        generated.push(next_token);

        if next_token == 151645 { break; } // EOS token for Qwen2.5
        last_token = next_token;
    }

    Ok(generated)
}

/// Generate tokens with slow-friend hook (observational mode)
fn generate_with_sf_hook(
    model: &mut pesti_runner::transformer::LlamaModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    sf_state: &mut SlowFriendState,
    divergence_scores: &mut Vec<f32>,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let mut generated = Vec::new();
    let mut last_token = prompt_tokens.last().copied().unwrap_or(0);

    // Forward pass on prompt with slow-friend tracking
    model.reset_cpu_kv_caches();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i)?;
        let h = model.forward_layers_with_cache(&hidden, i)?;

        // Slow-friend hook: update EMA and compute divergence
        sf_state.update(&h);
        let score = divergence(DivergenceMetric::Cosine, sf_state.summary(), &h);
        divergence_scores.push(score.value);

        if i == prompt_tokens.len() - 1 {
            last_token = tok;
        }
    }

    // Generate tokens with slow-friend tracking
    for step in 0..max_tokens {
        let hidden = model.embed(last_token, prompt_tokens.len() + step)?;
        let h = model.forward_layers_with_cache(&hidden, prompt_tokens.len() + step)?;

        // Slow-friend hook
        sf_state.update(&h);
        let score = divergence(DivergenceMetric::Cosine, sf_state.summary(), &h);
        divergence_scores.push(score.value);

        let logits = model.apply_output_head(&h)?;
        let next_token = argmax(&logits);
        generated.push(next_token);

        if next_token == 151645 { break; } // EOS token for Qwen2.5
        last_token = next_token;
    }

    Ok(generated)
}

/// Generate with perturbation applied after step N to induce drift
fn generate_with_perturbation(
    model: &mut pesti_runner::transformer::LlamaModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    sf_state: &mut SlowFriendState,
    perturbed_divergences: &mut Vec<(usize, f32)>,
    perturbation_step: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut last_token = prompt_tokens.last().copied().unwrap_or(0);

    // Forward pass on prompt
    model.reset_cpu_kv_caches();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i)?;
        let h = model.forward_layers_with_cache(&hidden, i)?;
        sf_state.update(&h);
        if i == prompt_tokens.len() - 1 {
            last_token = tok;
        }
    }

    // Generate with perturbation
    for step in 0..max_tokens {
        let hidden = model.embed(last_token, prompt_tokens.len() + step)?;
        let h = model.forward_layers_with_cache(&hidden, prompt_tokens.len() + step)?;

        // Apply perturbation to hidden state after specified step
        let mut perturbed_h = h.clone();
        if step >= perturbation_step && step < perturbation_step + 10 {
            for v in perturbed_h.iter_mut() {
                *v *= 1.05;
            }
        }

        sf_state.update(&perturbed_h);
        let score = divergence(DivergenceMetric::Cosine, sf_state.summary(), &perturbed_h);
        perturbed_divergences.push((step, score.value));

        // Use unperturbed logits for token selection (perturbation only affects hidden state tracking)
        let logits = model.apply_output_head(&h)?;
        let next_token = argmax(&logits);

        if next_token == 151645 { break; } // EOS token for Qwen2.5
        last_token = next_token;
    }

    Ok(())
}

/// Argmax sampling (deterministic)
fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}
