//! Minimal test of pesti-runner generation
use pesti_runner::llama::{LlamaRunner, SamplingConfig};

fn main() {
    let model_path = std::env::args().nth(1).expect("Usage: test_gen <model.gguf>");
    
    println!("Loading model from: {}", model_path);
    let runner = LlamaRunner::builder(&model_path)
        .n_ctx(2048)
        .build()
        .expect("Failed to build runner");

    println!("Model loaded. Generating...");
    
    let config = SamplingConfig {
        temperature: 0.7,
        top_k: 50,
        top_p: 0.9,
        max_tokens: 20,
        seed: 42,
        ..Default::default()
    };

    match runner.generate("What is 2+2?", &config) {
        Ok(result) => {
            println!("Generated text: {}", result.text);
            println!("Tokens generated: {}", result.generated_tokens);
        }
        Err(e) => {
            eprintln!("Generation failed: {}", e);
            std::process::exit(1);
        }
    }
}
