//! Long-sequence benchmark to measure where batched/chunked generation provides value.
//! Tests prefill time and decode speed at increasing context lengths.

use std::path::Path;
use std::time::Instant;

fn main() {
    let model_path = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    if !Path::new(model_path).exists() {
        eprintln!("Model not found: {}", model_path);
        std::process::exit(1);
    }

    println!("=== PESTI Long Sequence Benchmark ===");
    println!("Model: qwen2.5-0.5b-instruct-q4_k_m (Q4_K_M)");
    println!();

    // Load model once
    let t_load = Instant::now();
    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))
        .expect("Failed to load GGUF weights");
    let mut model = pesti_runner::LlamaModel::from_gguf_weights(weights)
        .expect("Failed to build model");
    println!("Model loaded in {:.2}s", t_load.elapsed().as_secs_f64());

    // Load tokenizer
    let tokenizer_config = pesti_runner::transformer::GgufTokenizerConfig {
        vocab_size: 152064,
        pad_token_id: Some(151663),
        bos_token_id: Some(151663),
        eos_token_id: Some(151663),
    };
    let tokenizer = pesti_runner::transformer::load_tokenizer_from_gguf(
        Path::new(model_path), tokenizer_config
    ).expect("Failed to load tokenizer");
    model.tokenizer = Some(Box::new(tokenizer));

    // Test at different context lengths
    let seq_lengths = [128, 256, 512, 1024, 2048];
    
    for &seq_len in &seq_lengths {
        println!("--- Sequence length: {} ---", seq_len);
        
        // Create synthetic long prompt (repeated pattern)
        let base_prompt = "The quick brown fox jumps over the lazy dog. ";
        let mut long_prompt = String::new();
        while long_prompt.len() < seq_len * 4 {  // ~4 chars per token avg
            long_prompt.push_str(base_prompt);
        }
        
        // Encode
        let t_encode = Instant::now();
        let tokens = tokenizer.encode(&long_prompt)
            .expect("Failed to encode")
            .into_iter()
            .take(seq_len)  // Truncate to exact length
            .collect::<Vec<u32>>();
        println!("Encoded {} tokens in {:.3}s", tokens.len(), t_encode.elapsed().as_secs_f64());
        
        // Prefill (first forward pass with full sequence)
        let sampling = pesti_runner::transformer::SamplingConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: Some(42),
        };
        
        let t_prefill = Instant::now();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // Generate just 1 token to measure prefill time
        let _result = model.generate(&tokens, 1, &sampling, &mut rng, &[151663])
            .expect("Generation failed");
        let prefill_time = t_prefill.elapsed().as_secs_f64();
        
        println!("Prefill time: {:.3}s ({:.0} tok/s effective)", 
            prefill_time, seq_len as f64 / prefill_time);
        println!();
    }

    println!("=== Benchmark Complete ===");
}
