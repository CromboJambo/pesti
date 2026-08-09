//! Simple CPU-only generation example using pesti-runner.
//! 
//! This demonstrates basic text generation with the Qwen2.5 0.5B model,
//! comparing pure-Rust inference against llama.cpp.

use pesti_runner::CpuModel;
use std::path::Path;
use std::time::Instant;

fn main() {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    println!("Loading model from: {}", model_path);
    let start = Instant::now();
    
    match CpuModel::load_gguf(Path::new(model_path)) {
        Ok(model) => {
            let load_time = start.elapsed();
            println!("✓ Model loaded in {:.2}s", load_time.as_secs_f32());
            println!("  - Hidden size: {}", model.hidden_size);
            println!("  - Vocab size: {}", model.vocab_size);
            
            // Test encoding/decoding
            let prompt = "Quantum computing";
            println!("\nEncoding prompt: '{}'", prompt);
            
            match model.encode(prompt) {
                Ok(tokens) => {
                    if tokens.is_empty() {
                        println!("  ⚠ Stub tokenizer returned no tokens (CPU-only mode)");
                        println!("\nSkipping token-based tests in stub mode.");
                    } else {
                        println!("  → {} tokens: {:?}", tokens.len(), tokens);

                        println!("\nDecoding back:");
                        match model.decode_tokens(&tokens) {
                            Ok(decoded) => println!("  → '{}'", decoded),
                            Err(e) => println!("  ⚠ Decode error: {}", e),
                        }

                        // Test embedding + output head (minimal inference)
                        if let Some(embeddings) = &model.token_embeddings {
                            println!("\nTesting minimal inference (embed → output):");

                            match model.embed(tokens[0], 0) {
                                Ok(hidden) => {
                                    println!(
                                        "  - Embedding token {} → hidden dim {}",
                                        tokens[0],
                                        hidden.len()
                                    );

                                    match model.apply_output_head(&hidden) {
                                        Ok(logits) => {
                                            println!(
                                                "  - Output logits: {} vocab dimensions",
                                                logits.len()
                                            );

                                            // Argmax to get next token
                                            let next_token = pesti_runner::argmax(&logits);
                                            println!("  - Next token (argmax): {}", next_token);

                                            match model.decode_tokens(&[next_token]) {
                                                Ok(next_text) => {
                                                    println!("  - Decoded: '{}'", next_text)
                                                }
                                                Err(e) => println!("  ⚠ Decode error: {}", e),
                                            }
                                        }
                                        Err(e) => println!("  ⚠ Output head error: {}", e),
                                    }
                                }
                                Err(e) => println!("  ⚠ Embedding error: {}", e),
                            }
                        } else {
                            println!("\n⚠ Token embeddings not available (CPU-only stub mode)");
                        }
                    }
                }
                Err(e) => println!("  ⚠ Encode error: {}", e),
            }
        }
        Err(e) => {
            eprintln!("✗ Failed to load model: {}", e);
            std::process::exit(1);
        }
    }
}
