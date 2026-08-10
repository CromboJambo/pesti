//! Quick verification that GPU forward_with_dispatch() produces logits (not hidden state)

use pesti_runner::transformer::LlamaModel;
use std::path::Path;

fn main() {
    let model_path = Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    );

    println!("=== GPU Output Head Verification ===\n");

    // Load model with dispatch context
    let mut model = match LlamaModel::load_gguf(model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to load: {}", e);
            return;
        }
    };

    println!("[OK] Loaded Qwen2.5-0.5B");
    println!("  - Architecture: {:?}", model.config.arch);
    println!("  - Hidden dim: {}", model.config.embed_dim);
    println!("  - Vocab size: {}", model.vocab_size);
    println!("  - Layers: {}", model.config.num_layers);

    // Create test input (hidden state from middle of vocab)
    let hidden_size = model.config.embed_dim;
    let test_input: Vec<f32> = (0..hidden_size)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();

    println!("\n[TEST] Testing GPU forward_with_dispatch()...");

    match model.forward_with_dispatch(&test_input, 0) {
        Ok(output) => {
            println!("[OK] Output produced: {} elements", output.len());
            
            // Expected: vocab_size logits (~32k for Qwen2.5-0.5B)
            if output.len() == model.vocab_size as usize {
                println!("[PASS] Output length = vocab_size");
                println!("  First 5 values: {:.6}, {:.6}, {:.6}, {:.6}, {:.6}", 
                    output[0], output[1], output[2], output[3], output[4]);
            } else {
                eprintln!("[FAIL] Output length {} != vocab_size {}", output.len(), model.vocab_size);
                println!("  This suggests forward_with_dispatch() is returning hidden state, not logits");
            }
        },
        Err(e) => {
            eprintln!("[ERROR] Forward failed: {}", e);
            
            // Check if it's a missing output head error (our new guard)
            if e.to_string().contains("missing output layer") {
                println!("  -> Output weight tensor not loaded from GGUF");
            } else if e.to_string().contains("dispatch context") {
                println!("  -> Need to initialize dispatch properly");
            }
        }
    }
}
