//! End-to-end test: Safetensors loading + tokenizers crate inference.
//!
//! Proves the full Rust inference path works without llama-cpp-sys-2:
//! 1. Parse GGUF header for config (pesti-gguf, pure Rust)
//! 2. Convert GGUF → Safetensors (pesti-safetensors, pure Rust)
//! 3. Load model from Safetensors (LlamaModel::load_safetensors)
//! 4. Tokenize with `tokenizers` crate (HF-compatible, no C++)
//! 5. Generate tokens via pure Rust forward pass
//! 6. Decode output
//!
//! This test validates that pesti can drop llama-cpp-sys-2 as a hard dependency.

use pesti_runner::transformer::{LlamaConfig, SamplingConfig, LlamaModel, load_tokenizer_from_gguf, TokenizerBackend};
use pesti_gguf::parse_gguf;
use pesti_safetensors::gguf_converter::convert_gguf_to_safetensors;
use rand::SeedableRng;
use std::path::Path;
use std::time::Instant;

#[test]
fn test_e2e_safetensors_inference() {
    // Use the conformance corpus model (Qwen2.5-0.5B, Q4_K_M)
    let gguf_path = Path::new(
        "/home/crombo/projects/active/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );

    if !gguf_path.exists() {
        eprintln!("Skipping: GGUF model not found at {:?}", gguf_path);
        return;
    }

    let output_dir = std::env::temp_dir().join("pesti-e2e-safetensors");
    std::fs::create_dir_all(&output_dir).expect("Failed to create temp dir");

    println!("=== E2E Safetensors Inference Test (no llama.cpp) ===");
    println!();

    // Step 0: Parse GGUF header to extract config (pure Rust, no FFI)
    println!("Step 0: Parsing GGUF header for config...");
    let gguf_header = parse_gguf(gguf_path).expect("Failed to parse GGUF header");
    let config = LlamaConfig::from_gguf_header(&gguf_header).expect("Failed to extract config");
    println!("  Config: arch={:?}, layers={}, dim={}, heads={}", 
        config.arch, config.num_layers, config.embed_dim, config.num_heads);
    println!();

    // Step 1: Convert GGUF → Safetensors (pure Rust)
    println!("Step 1: Converting GGUF → Safetensors...");
    let t_convert = Instant::now();
    let result = convert_gguf_to_safetensors(gguf_path, &output_dir)
        .expect("Failed to convert GGUF to Safetensors");
    println!(
        "  Converted {} tensors ({:.2} MB) in {:.2}s",
        result.tensor_count,
        result.total_bytes as f64 / 1_048_576.0,
        t_convert.elapsed().as_secs_f64()
    );

    // Find the output safetensors file
    let safetensors_path = std::fs::read_dir(&output_dir)
        .expect("Failed to read output dir")
        .filter_map(|e| e.ok())
        .find(|e| e.path().extension().map_or(false, |ext| ext == "safetensors"))
        .expect("No safetensors file found")
        .path();

    println!("  Output: {:?}", safetensors_path);
    println!();

    // Step 2: Load model from Safetensors (pure Rust - no llama.cpp FFI)
    println!("Step 2: Loading model from Safetensors...");
    let t_load = Instant::now();
    let mut model = LlamaModel::load_safetensors(&safetensors_path, config.clone())
        .expect("Failed to load model from Safetensors");
    println!(
        "  Loaded in {:.2}s (dim={}, layers={})",
        t_load.elapsed().as_secs_f64(),
        model.config.embed_dim,
        model.config.num_layers
    );
    println!();

    // Step 3: Load tokenizer using `tokenizers` crate (HF-compatible)
    println!("Step 3: Loading tokenizer via `tokenizers` crate...");
    let t_tok_load = Instant::now();
    let (_config, tokenizer) = load_tokenizer_from_gguf(gguf_path, TokenizerBackend::MistralRs)
        .expect("Failed to load tokenizer");
    let vocab_size = tokenizer.vocab_size();
    model.tokenizer = Some(tokenizer);
    println!(
        "  Loaded in {:.3}s (vocab_size={})",
        t_tok_load.elapsed().as_secs_f64(),
        vocab_size
    );
    println!();

    // Step 4: Tokenize prompt with `tokenizers` crate
    let prompt = "What is Rust?";
    println!("Step 4: Tokenizing prompt: \"{}\"", prompt);
    let tok = model.tokenizer.as_ref().expect("No tokenizer");
    let prompt_tokens = tok.encode(prompt).expect("Failed to encode");
    println!(
        "  Encoded {} tokens: {:?}",
        prompt_tokens.len(),
        &prompt_tokens[..std::cmp::min(5, prompt_tokens.len())]
    );
    println!();

    // Step 5: Generate tokens (pure Rust inference)
    let max_tokens = 32;
    let sampling = SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: Some(42),
    };

    println!("Step 5: Generating {} tokens (greedy, pure Rust)...", max_tokens);
    let t_gen = Instant::now();
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let generated_tokens = model
        .generate(&prompt_tokens, max_tokens, &sampling, &mut rng, &[151663])
        .expect("Generation failed");
    let gen_time = t_gen.elapsed().as_secs_f64();
    println!(
        "  Generated {} tokens in {:.2}s ({:.1} tok/s)",
        generated_tokens.len(),
        gen_time,
        if gen_time > 0.0 {
            generated_tokens.len() as f64 / gen_time
        } else {
            0.0
        }
    );
    println!();

    // Step 6: Decode output with `tokenizers` crate
    let tok = model.tokenizer.as_ref().expect("No tokenizer");
    let generated_text = tok.decode(&generated_tokens).expect("Failed to decode");
    println!("Step 6: Decoded output:");
    println!("  \"{}\"", generated_text);
    println!();

    // Assertions: verify we actually got meaningful output
    assert!(
        !generated_tokens.is_empty(),
        "Generated zero tokens — inference failed"
    );
    assert!(
        generated_text.chars().count() > 5,
        "Generated text too short ({}) — likely garbage",
        generated_text.chars().count()
    );

    // Cleanup
    let _ = std::fs::remove_dir_all(&output_dir);

    println!("=== E2E TEST PASSED ===");
    println!(
        "Full Rust path (Safetensors + tokenizers crate) works without llama-cpp-sys-2."
    );
}
