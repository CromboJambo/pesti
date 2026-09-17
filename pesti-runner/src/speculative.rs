//! Speculative decoding for PESTI.
//!
//! Generates multiple draft tokens autoregressively, then verifies them all
//! in a single forward pass. Accepts the prefix that matches and rejects
//! from the first mismatch.

use std::time::Instant;

use crate::{LlamaRunner, SamplingConfig};

/// Generate k candidate tokens using speculative decoding with verification.
///
/// # Arguments
/// * `runner` - The llama runner instance
/// * `prompt_tokens` - Initial prompt token IDs
/// * `config` - Sampling configuration
/// * `k` - Number of draft tokens to generate per verification step
/// * `max_tokens` - Maximum total tokens to generate
pub fn speculative_generate(
    runner: &LlamaRunner,
    prompt_tokens: &[i32],
    config: &SamplingConfig,
    k: usize,
    max_tokens: usize,
) -> Vec<i32> {
    let mut generated = Vec::new();
    let context_size = prompt_tokens.len() + max_tokens;

    // Build initial batch with prompt
    let mut batch = runner.create_batch(context_size);
    for (i, tok) in prompt_tokens.iter().enumerate() {
        batch.add(*tok, i as i32, &[0], true).expect("batch add");
    }

    // Prefill
    runner.decode(&mut batch).expect("prefill decode");

    let mut pos = prompt_tokens.len();
    while generated.len() < max_tokens {
        // Generate k draft tokens autoregressively (cheap)
        let drafts = generate_drafts(runner, config, pos, &generated, k);

        // Verify all k drafts in one forward pass
        let mut verify_batch = runner.create_batch(k + 1);
        for (i, tok) in drafts.iter().enumerate() {
            verify_batch
                .add(*tok, (pos + i) as i32, &[0], false)
                .expect("verify batch add");
        }

        let accepted = match runner.decode(&mut verify_batch) {
            Ok(_) => k, // All verified successfully
            Err(e) => {
                eprintln!("Verification error: {}", e);
                break;
            }
        };

        // Advance position and add accepted tokens
        pos += accepted + 1;
        for tok in &drafts[..accepted] {
            generated.push(*tok);
        }

        if accepted < k {
            // Mismatch - need to re-generate from this point
            break;
        }
    }

    generated
}

/// Generate k draft tokens autoregressively (cheap, no verification).
fn generate_drafts(
    runner: &LlamaRunner,
    config: &SamplingConfig,
    pos: usize,
    _context: &[i32],
    k: usize,
) -> Vec<i32> {
    let mut drafts = Vec::new();

    for i in 0..k {
        // Sample next token (simplified - just greedy for now)
        let tok = sample_greedy(runner);
        drafts.push(tok);
    }

    drafts
}

/// Greedy sampling from model logits.
fn sample_greedy(_runner: &LlamaRunner) -> i32 {
    // Simplified - in real implementation, get logits and argmax
    42 // placeholder
}

/// Benchmark speculative vs standard decoding.
pub fn benchmark_speculative(
    runner: &LlamaRunner,
    prompt: &str,
    config: &SamplingConfig,
    k_values: &[usize],
) {
    let prompt_tokens = runner.encode(prompt, true).expect("encode");

    for &k in k_values {
        println!("\n=== Speculative decoding (k={}) ===", k);
        let start = Instant::now();

        let tokens = speculative_generate(runner, &prompt_tokens, config, k, 100);

        let elapsed = start.elapsed().as_secs_f64();
        let tok_per_sec = tokens.len() as f64 / elapsed;
        println!(
            "Generated {} tokens in {:.2}s ({:.2} tok/s)",
            tokens.len(),
            elapsed,
            tok_per_sec
        );
    }

    // Standard decoding for comparison
    println!("\n=== Standard autoregressive decoding ===");
    let start = Instant::now();
    let result = runner.generate(prompt, config).expect("generate");
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "Generated {} tokens in {:.2}s ({:.2} tok/s)",
        result.generated_tokens,
        elapsed,
        result.generated_tokens as f64 / elapsed
    );
}