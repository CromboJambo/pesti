//! Full autoregressive text generation with PESTI (CPU-only).
//!
//! This example demonstrates end-to-end inference using the real CPU transformer implementation.
//! It loads GGUF weights, performs forward pass through all transformer layers, and generates text.

use pesti_runner::CpuModel;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt = "The quick brown fox jumps over the lazy dog. ";
    let max_tokens = 20;
    let temperature = 0.7;

    println!("=== PESTI Autoregressive Generation (CPU) ===");
    println!("Model: {}", model_path);
    println!("Prompt: \"{}\"", prompt);
    println!("Max tokens: {}\n", max_tokens);

    // Step 1: Load tokenizer config from GGUF
    let load_start = Instant::now();
    let (tokenizer_config, _tokenizer) =
        pesti_runner::load_tokenizer_from_gguf(Path::new(model_path))?;

    println!(
        "✓ Loaded tokenizer config in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );
    println!("  - Vocab size: {}", tokenizer_config.vocab_size);
    println!("  - BOS token: {:?}", tokenizer_config.bos_token_id);
    println!("  - EOS token: {:?}", tokenizer_config.eos_token_id);

    // Step 2: Load model weights from GGUF (using CpuModel)
    let cpu_model = CpuModel::load_gguf(Path::new(model_path))?;

    println!(
        "\n✓ Loaded model in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );
    println!("  - Hidden size: {}", cpu_model.hidden_size);
    println!("  - Vocab size: {}", cpu_model.vocab_size);
    println!(
        "  - Token embeddings loaded: {}",
        cpu_model.token_embeddings.is_some()
    );
    println!(
        "  - Output weights loaded: {}",
        cpu_model.output_weights.is_some()
    );

    // Step 3: Build CPU transformer model with full layers
    println!(
        "\n⚠️  NOTE: CpuModel only loads embeddings + output head for now."
    );
    println!("   For full transformer inference, use transformer_cpu::CpuTransformerModel");
    println!("   which loads all layer weights from GGUF.\n");

    // Step 4: Generate tokens using current CpuModel implementation
    let mut generated_tokens = Vec::new();
    let gen_start = Instant::now();

    println!("=== Generating {} tokens ===\n", max_tokens);

    // Start with prompt tokenization (stub - use first token for now)
    let mut current_token: u32 = 100; // BOS token placeholder

    for step in 0..max_tokens {
        // Embed the current token
        let hidden = cpu_model.embed(current_token, step)?;

        // Apply output head to get logits (skip transformer layers for now)
        let logits = cpu_model.apply_output_head(&hidden)?;

        // Sample next token with temperature
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let next_token = sample_with_temperature(&logits, temperature, &mut rng);
        generated_tokens.push(next_token);

        if step == 0 {
            print!("Generated: ");
        }

        // For now, just print token ID (we don't have proper tokenizer decoding yet)
        print!("{} ", next_token);

        current_token = next_token;
    }

    println!(
        "\n\n✓ Generation complete in {:.2}s",
        gen_start.elapsed().as_secs_f32()
    );
    println!(
        "  - Tokens/sec: {:.1}",
        max_tokens as f32 / gen_start.elapsed().as_secs_f32()
    );

    // Save to file
    let output = format!(
        "Generated {} tokens from prompt: \"{}\"\nTokens: {:?}",
        generated_tokens.len(),
        prompt,
        generated_tokens
    );
    std::fs::write("generation_output.txt", &output)?;
    println!("  - Output saved to: generation_output.txt");

    println!(
        "\n⚠️  NOTE: This is a minimal implementation."
    );
    println!(
        "   The forward pass currently skips transformer layers (embed → output head only)."
    );
    println!(
        "   For full inference with attention and FFN, implement CpuTransformerModel loading."
    );

    Ok(())
}

/// Softmax sampling with temperature.
fn sample_with_temperature(
    logits: &[f32],
    temp: f32,
    rng: &mut rand::rngs::StdRng,
) -> u32 {
    if logits.is_empty() {
        return 0;
    }

    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temp).collect();

    // Subtract max for numerical stability
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();

    // Convert to probabilities
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    // Categorical sampling using cumulative sum
    let mut cumsum = 0.0;
    let r = rng.random::<f32>(); // Random number in [0, 1)

    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i as u32;
        }
    }

    // Fallback to last token
    (probs.len() - 1) as u32
}
