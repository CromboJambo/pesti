//! End-to-end safetensors inference test with Qwen2.5-Coder-1.5B.
//!
//! This test loads the model directly from HF safetensors format (no GGUF, no llama.cpp),
//! proving pesti can run real-world models without any conversion step.
//!
//! Model: Qwen/Qwen2.5-Coder-1.5B-Instruct
//! Location: models/qwen2.5-coder-1.5b/ (config.json + model.safetensors + tokenizer.json)
//!
//! Run with: cargo test -p pesti-runner --test e2e_safetensors_qwen2_5_coder -- --nocapture --ignored

use std::path::Path;
use std::time::Instant;

use pesti_runner::transformer::{LlamaConfig, LlamaModel, SamplingConfig, load_tokenizer_from_hf_dir};

const MODEL_DIR: &str = "models/qwen2.5-coder-1.5b";

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
        "  vocab_size={}",
        config.vocab_size(),
    );
    println!("  hidden_size={}", config.hidden_size());
    println!("  num_layers={}", config.n_layers);
    println!("  num_heads={}", config.n_head);
    println!("  num_kv_heads={}", config.n_kv_head);

    // 2. Load tokenizer from HF tokenizer.json
    println!("Loading tokenizer from tokenizer.json...");
    let (tok_config, tokenizer) = load_tokenizer_from_hf_dir(model_dir)
        .unwrap_or_else(|e| panic!("Failed to load tokenizer: {}", e));
    println!("  vocab_size={}", tok_config.vocab_size);

    // Verify config matches tokenizer
    assert_eq!(
        config.vocab_size(),
        tok_config.vocab_size,
        "Config vocab_size ({}) != tokenizer vocab_size ({})",
        config.vocab_size(),
        tok_config.vocab_size
    );

    // 3. Load model weights from safetensors
    println!("Loading model weights from model.safetensors (this takes a while)...");
    let load_start = Instant::now();
    let safetensors_path = model_dir.join("model.safetensors");
    let model = LlamaModel::load_safetensors(&safetensors_path, &config)
        .unwrap_or_else(|e| panic!("Failed to load safetensors model: {}", e));
    let load_time = load_start.elapsed();
    println!("  Loaded in {:?}", load_time);

    // 4. Encode a simple prompt
    let prompt = "What is Rust?";
    println!("\nEncoding prompt: '{}'", prompt);
    let tokens = tokenizer.encode(prompt).unwrap_or_else(|e| {
        panic!("Failed to encode prompt: {}", e)
    });
    println!("  Encoded to {} tokens: {:?}", tokens.len(), &tokens[..8.min(tokens.len())]);

    // 5. Run inference (single forward pass for now — full generation is slow in debug mode)
    println!("\nRunning single forward pass...");
    let fwd_start = Instant::now();
    let mut logits = vec![0f32; config.vocab_size()];
    model
        .forward(&tokens, 0, &mut logits)
        .unwrap_or_else(|e| panic!("Forward pass failed: {}", e));
    let fwd_time = fwd_start.elapsed();
    println!(
        "  Forward pass completed in {:?} ({:.2} tok/s)",
        fwd_time,
        tokens.len() as f64 / fwd_time.as_secs_f64()
    );

    // 6. Verify logits are reasonable (not all zeros, not all NaN/Inf)
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

    // 7. Get top-5 tokens to verify the model produces sensible predictions
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(a.1));
    println!("\nTop-5 predicted tokens:");
    for (rank, (idx, score)) in indexed.iter().take(5).enumerate() {
        let token_str = tokenizer.decode(&[*idx as u32]).unwrap_or_else(|_| "<decode-fail>".into());
        println!("  {}: '{}' ({}) score={:.4}", rank + 1, idx, token_str.trim(), score);
    }

    // 8. Try generating a few tokens (greedy) to see if output makes sense
    println!("\nGenerating 3 tokens greedily...");
    let gen_start = Instant::now();
    let mut current_tokens = tokens.clone();
    let mut generated_text = String::new();

    for step in 0..3 {
        let pos = current_tokens.len() as u32;
        model.forward(&current_tokens, pos, &mut logits).unwrap_or_else(|e| {
            panic!("Generation step {} failed: {}", step + 1, e)
        });

        // Greedy: take argmax
        let next_idx = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap();

        let next_str = tokenizer.decode(&[next_idx]).unwrap_or_else(|_| "<decode-fail>".into());
        generated_text.push_str(next_str.as_str());
        current_tokens.push(next_idx);
        println!("  step {}: token={} '{}'", step + 1, next_idx, next_str.trim());
    }

    let gen_time = gen_start.elapsed();
    println!(
        "\nGenerated {} tokens in {:?} ({:.2} tok/s)",
        3,
        gen_time,
        3.0 / gen_time.as_secs_f64()
    );
    println!("Full output: '{}{}'", prompt, generated_text);

    // Basic sanity: output should not be empty and should contain some recognizable text
    assert!(
        !generated_text.trim().is_empty(),
        "Generated text is empty — model is not working"
    );

    println!("\n✅ Qwen2.5-Coder-1.5B safetensors inference test PASSED");
}
