//! End-to-end safetensors inference test with Qwen2.5-Coder-1.5B.
//!
//! This test loads the model directly from HF safetensors format (no GGUF, no llama.cpp),
//! proving pesti can run real-world models without any conversion step.
//!
//! Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
//! Location: models/qwen2.5-coder-1.5b/ (config.json + model.safetensors + tokenizer.json)
//!
//! Run with: cargo test --release -p pesti-runner --test e2e_safetensors_qwen2_5_coder -- --ignored --nocapture

use std::path::Path;
use std::time::Instant;

use pesti_runner::transformer::{load_tokenizer_from_hf_dir, LlamaConfig, LlamaModel, SamplingConfig};
use rand::SeedableRng;

const MODEL_DIR: &str = "../../models/qwen2.5-coder-1.5b";

#[test]
#[ignore] // 1.5B model is slow in debug mode; run with --ignored when needed
fn test_qwen2_5_coder_safetensors_inference() {
    let model_dir = Path::new(MODEL_DIR);
    assert!(model_dir.exists(), "Model directory not found: {}", MODEL_DIR);

    // 1. Load config from HF config.json
    println!("Loading config from config.json...");
    let config_path = model_dir.join("config.json");
    let config = LlamaConfig::from_hf_config_json(&config_path)
        .unwrap_or_else(|e| panic!("Failed to load config: {}", e));

    println!(
        "  embed_dim={}, layers={}, heads={}, kv_heads={}",
        config.embed_dim,
        config.num_layers,
        config.num_heads,
        config.num_kv_heads
    );

    // 2. Load tokenizer from HF tokenizer.json
    println!("Loading tokenizer from tokenizer.json...");
    let (tok_config, mut tokenizer) = load_tokenizer_from_hf_dir(model_dir)
        .unwrap_or_else(|e| panic!("Failed to load tokenizer: {}", e));
    println!("  vocab_size={}", tok_config.vocab_size);

    // 3. Load model weights from safetensors
    println!("Loading model weights from model.safetensors...");
    let load_start = Instant::now();
    let mut model = LlamaModel::load_safetensors(&model_dir.join("model.safetensors"), config)
        .unwrap_or_else(|e| panic!("Failed to load safetensors model: {}", e));
    let load_time = load_start.elapsed();
    println!(
        "  Loaded in {:?} (vocab_size={})",
        load_time, model.vocab_size
    );

    // Verify vocab size matches tokenizer (model may pad embeddings, but not shrink)
    assert!(
        model.vocab_size as usize >= tok_config.vocab_size,
        "Model vocab_size ({}) < tokenizer vocab_size ({})",
        model.vocab_size,
        tok_config.vocab_size
    );

    // Attach tokenizer for generation
    model.tokenizer = Some(tokenizer);

    // 4. Encode a simple prompt
    let prompt = "What is Rust?";
    println!("\nEncoding prompt: '{}'", prompt);
    let tokens = model
        .tokenizer
        .as_ref()
        .unwrap()
        .encode(prompt)
        .unwrap_or_else(|e| panic!("Failed to encode prompt: {}", e));
    println!(
        "  Encoded to {} tokens: {:?}",
        tokens.len(),
        &tokens[..8.min(tokens.len())]
    );

    // 5. Single forward pass to verify logits are reasonable
    println!("\nRunning single forward pass on first token...");
    let fwd_start = Instant::now();
    let logits = model
        .forward(tokens[0], 0)
        .unwrap_or_else(|e| panic!("Forward pass failed: {}", e));
    let fwd_time = fwd_start.elapsed();
    println!(
        "  Forward pass completed in {:?} ({:.2} tok/s)",
        fwd_time,
        1.0 / fwd_time.as_secs_f64()
    );

    // Verify logits are reasonable (not all zeros, not all NaN/Inf)
    let finite_count = logits.iter().filter(|v| v.is_finite()).count();
    let non_zero_count = logits.iter().filter(|v| **v != 0.0f32).count();
    println!(
        "  Logits: {} finite, {} non-zero (total {})",
        finite_count, non_zero_count, logits.len()
    );

    assert!(
        finite_count > logits.len() / 2,
        "Too many non-finite logits — model output is garbage"
    );
    assert!(
        non_zero_count > 1000,
        "Too many zero logits — model weights may not be loaded correctly"
    );

    // Get top-5 tokens to verify the model produces sensible predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("\nTop-5 predicted tokens after first prompt token:");
    for (rank, (idx, score)) in indexed.iter().take(5).enumerate() {
        let token_str = model
            .tokenizer
            .as_ref()
            .unwrap()
            .decode(&[*idx as u32])
            .unwrap_or_else(|_| "<decode-fail>".into());
        println!(
            "  {}: '{}' ({}) score={:.4}",
            rank + 1,
            idx,
            token_str.trim(),
            score
        );
    }

    // 6. Full generation via generate() (greedy)
    let max_tokens = 32;
    println!("\nGenerating {} tokens greedily...", max_tokens);
    let sampling = SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: Some(42),
    };
    let gen_start = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let generated_tokens = model
        .generate(&tokens, max_tokens, &sampling, &mut rng, &[151663])
        .unwrap_or_else(|e| panic!("Generation failed: {}", e));
    let gen_time = gen_start.elapsed();
    println!(
        "  Generated {} tokens in {:?} ({:.2} tok/s)",
        generated_tokens.len(),
        gen_time,
        if gen_time.as_secs_f64() > 0.0 {
            generated_tokens.len() as f64 / gen_time.as_secs_f64()
        } else {
            0.0
        }
    );

    // Decode output
    let generated_text = model
        .tokenizer
        .as_ref()
        .unwrap()
        .decode(&generated_tokens)
        .unwrap_or_else(|e| panic!("Failed to decode: {}", e));
    println!("\nFull output:\n  '{}{}'", prompt, generated_text);

    // Basic sanity: output should not be empty and should contain some recognizable text
    assert!(
        !generated_tokens.is_empty(),
        "Generated zero tokens — inference failed"
    );
    assert!(
        generated_text.chars().count() > 5,
        "Generated text too short ({}) — likely garbage",
        generated_text.chars().count()
    );

    println!("\n✅ Qwen2.5-Coder-1.5B safetensors inference test PASSED");
}
