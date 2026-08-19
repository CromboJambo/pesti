//! Numerical conformance test for PESTI CPU inference path.
//!
//! Validates basic GGUF loading, model construction, and forward pass with
//! the stub implementation (no CUDA).

use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== PESTI Numerical Conformance Test (CPU Stub) ===");
    println!("Model: {}", model_path);

    // Load weights
    let t0 = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    println!("✅ Loaded weights in {:.2}s", t0.elapsed().as_secs_f32());
    println!(
        "   tensors={}, bytes={}",
        weights.tensors.len(),
        weights.tensors.values().map(|v| v.len()).sum::<usize>()
    );

    // Load model (CPU fallback uses transformer_stub)
    let t1 = Instant::now();
    let mut model = pesti_runner::LlamaModel::load_gguf(Path::new(model_path))?;
    println!("✅ Built model in {:.2}s", t1.elapsed().as_secs_f32());

    // Print model architecture
    println!("\n📐 Model Architecture:");
    println!("   hidden_size={}, embed_dim={}", model.hidden_size, model.embed_dim);
    println!("   num_layers={}, vocab_size={}", model.num_layers, model.vocab_size);
    println!("   num_heads={}, num_kv_heads={}, head_dim={}", model.num_heads, model.num_kv_heads, model.head_dim);
    println!("   rope_base={}, max_seq_len={}", model.rope_base, model.max_seq_len);

    // Load tokenizer (CPU fallback uses stub)
    let (tokenizer_config, tokenizer) = pesti_runner::load_tokenizer_from_gguf(Path::new(model_path))?;
    println!("\n✅ Loaded tokenizer (vocab={})", tokenizer_config.vocab_size);

    // Test prompt
    let prompt = "The quick brown fox jumps over the lazy dog.";
    println!("\n📝 Prompt: \"{}\"", prompt);

    // Tokenize
    let t2 = Instant::now();
    let prompt_tokens = tokenizer.encode(prompt, true).map_err(|e| e.to_string())?;
    println!(
        "✅ Encoded {} tokens in {:.3}ms",
        prompt_tokens.len(),
        t2.elapsed().as_secs_f32() * 1000.0
    );

    // Run single forward pass (stub implementation)
    let batch_size = 1;
    let input_dim = model.embed_dim;
    let test_input = vec![1.0f32; input_dim]; // Dummy embedding
    
    println!("\n🔬 Running forward pass test...");
    let t3 = Instant::now();
    
    // Test forward_layers (stub just passes through)
    let hidden_out = model.forward_layers(&test_input, batch_size)?;
    println!("✅ Forward pass completed in {:.3}ms", t3.elapsed().as_secs_f32() * 1000.0);
    println!("   Input shape: {}, Output shape: {}", input_dim, hidden_out.len());

    // Test forward_with_dispatch (stub returns zero logits)
    let t4 = Instant::now();
    let dummy_logits = model.forward_with_dispatch(&test_input, 0)?;
    println!("✅ Dispatch forward completed in {:.3}ms", t4.elapsed().as_secs_f32() * 1000.0);
    println!("   Logits shape: {}", dummy_logits.len());

    // Test sampling (stub uses deterministic sampling)
    let config = pesti_runner::SamplingConfig {
        seed: 42,
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
    };

    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    let t5 = Instant::now();
    let sampled_token = pesti_runner::sample(&dummy_logits, &config, &mut rng);
    println!("✅ Sampled token: {} in {:.3}ms", sampled_token, t5.elapsed().as_secs_f32() * 1000.0);

    // Decode the sampled token
    let decoded = tokenizer.decode(&[sampled_token], true).map_err(|e| e.to_string())?;
    println!("   Decoded: \"{}\"", decoded);

    println!("\n✅ Conformance test complete!");
    println!("   Next steps:");
    println!("   1. Compare logits against llama.cpp reference output");
    println!("   2. Verify KV cache values match expected precision");
    println!("   3. Test with CUDA backend if available (cargo run --features cuda)");

    Ok(())
}
