//! Profile pesti's pure Rust inference stack — per-step timing.

use std::path::Path;
use std::time::Instant;
use pesti_runner::transformer::{LlamaModel, SamplingConfig};
use rand::SeedableRng;

const MODEL_PATH: &str = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
const PROMPT: &str = "Write a short story about a robot learning to cook.";
const NUM_TOKENS: usize = 8;

fn main() {
    println!("=== pesti-runner Pure Rust Profile ===");

    let start = Instant::now();
    let mut model = LlamaModel::load_gguf(Path::new(MODEL_PATH)).unwrap();
    model.config.max_seq_len = 256;
    println!("Model loaded in {:.3}s", start.elapsed().as_secs_f64());

    let input_ids = {
        let tokenizer = model.tokenizer.as_ref().expect("Tokenizer not loaded");
        tokenizer.encode(PROMPT).unwrap()
    };
    println!("Prompt tokens: {}", input_ids.len());

    let sampling = SamplingConfig {
        temperature: 0.7,
        top_p: 0.95,
        top_k: 50,
        seed: Some(42),
    };

    // Manual loop to time each step
    let mut generated = Vec::new();
    let total_start = Instant::now();

    for i in 0..NUM_TOKENS {
        let step_start = Instant::now();

        if i == 0 {
            // Prefill + first token
            let sampling_cfg = &sampling;
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            generated = model.generate(&input_ids, 1, sampling_cfg, &mut rng, &[]).unwrap();
        } else {
            // Continue from previous token
            let prev_token = generated.last().unwrap();
            let sampling_cfg = &sampling;
            let mut rng = rand::rngs::StdRng::seed_from_u64(42 + i as u64);
            let cont = model.generate(&[*prev_token], 1, sampling_cfg, &mut rng, &[]).unwrap();
            generated.push(*cont.first().unwrap());
        }

        let step_time = step_start.elapsed().as_secs_f64();
        println!("Step {}: {:.3}s ({:.2} tok/s)", i + 1, step_time, 1.0 / step_time);
    }

    let total = total_start.elapsed().as_secs_f64();
    println!("\nTotal: {} tokens in {:.3}s = {:.2} tok/s", generated.len(), total, generated.len() as f64 / total);
}
