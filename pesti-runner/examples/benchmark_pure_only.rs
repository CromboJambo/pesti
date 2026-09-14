//! Benchmark: pesti's pure Rust inference stack (standalone).
//! Uses a small sequence length to fit in GPU memory.

use std::path::Path;
use std::time::Instant;
use pesti_runner::transformer::{LlamaModel, SamplingConfig};
use rand::SeedableRng;

const MODEL_PATH: &str = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const PROMPT: &str = "Write a short story about a robot learning to cook.";
const NUM_TOKENS: usize = 32;

fn main() {
    println!("=== pesti-runner Pure Rust Benchmark ===");
    println!("Model: Qwen2.5-0.5B-Instruct-Q4_K_M");
    println!("Hardware: RTX 4070 Ti SUPER (CUDA)");
    println!("Tokens to generate: {}", NUM_TOKENS);

    let start = Instant::now();

    // Load model using pesti's own GGUF loader + transformer implementation
    let mut model = LlamaModel::load_gguf(Path::new(MODEL_PATH)).unwrap();

    // Reduce max_seq_len to fit in GPU memory
    model.config.max_seq_len = 256;

    // Tokenize using pesti's tokenizer (loaded from GGUF)
    let input_ids = {
        let tokenizer = model.tokenizer.as_ref().expect("Tokenizer not loaded");
        tokenizer.encode(PROMPT).unwrap()
    };
    println!("Prompt tokens: {}", input_ids.len());

    // Generate tokens (pure Rust forward pass + pesti CUDA kernels)
    let sampling = SamplingConfig {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 50,
        seed: Some(42),
    };

    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let generated = model.generate(&input_ids, NUM_TOKENS, &sampling, &mut rng, &[]).unwrap();
    let elapsed = start.elapsed().as_secs_f64();

    println!("Generated {} tokens in {:.3}s", generated.len(), elapsed);
    let tok_per_sec = generated.len() as f64 / elapsed;
    println!("Throughput: {:.2} tok/s (pure Rust)", tok_per_sec);
    
    // Decode and print output
    let tokenizer = model.tokenizer.as_ref().expect("Tokenizer not loaded");
    let decoded = tokenizer.decode(&generated).unwrap();
    println!("
Generated text:
{}", decoded);
}
