//! Example demonstrating GPU-softmax with feature gating.
//!
//! This example shows how softmax can be computed on CPU or GPU depending on features.
//!
//! Usage (CPU-only): `cargo run --example softmax_example`
//! Usage (with CUDA): `cargo run --example softmax_example --features cuda`

fn main() {
    use pesti_runner::kernel::softmax::{SoftmaxKernel, SoftmaxKernelBuilder};

    // Sample logits for token sampling (vocabulary size = 1000)
    let logits = vec![2.0, 1.0, 0.0, -1.0, -2.0];

    println!("PESTI Softmax Example");
    println!("=====================");
    println!("\nInput logits: {:?}", logits);

    // Create softmax kernel based on available features
    let kernel = SoftmaxKernelBuilder::auto();

    println!("\nBackend: {}", kernel.name());
    println!("Available: {}", kernel.is_available());

    // Compute softmax
    match kernel.forward(&logits) {
        Ok(probs) => {
            println!("\nSoftmax probabilities:");
            for (i, &p) in probs.iter().enumerate() {
                println!("  Token {}: {:.6}", i, p);
            }

            let sum: f32 = probs.iter().sum();
            println!("\nSum of probabilities: {:.10} (should be ~1.0)", sum);

            // Show top-k tokens
            let k = 3;
            let mut indexed_probs: Vec<(f32, usize)> = probs.iter().enumerate().map(|(i, &p)| (p, i)).collect();
            indexed_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            
            println!("\nTop-{} tokens:", k);
            for &(p, idx) in indexed_probs.iter().take(k) {
                println!("  Token {} with probability {:.4}", idx, p);
            }
        }
        Err(e) => {
            eprintln!("Error computing softmax: {}", e);
        }
    }

    // Demonstrate numerical stability with large values
    println!("\n\nNumerical Stability Test");
    println!("========================");
    let large_logits = vec![1000.0, 1001.0, 1002.0];
    println!("\nLarge logits (would overflow without max subtraction): {:?}", large_logits);

    match kernel.forward(&large_logits) {
        Ok(probs) => {
            println!("Softmax (stable): {:?}", probs);
            let sum: f32 = probs.iter().sum();
            println!("Sum: {:.10} (should be ~1.0)", sum);
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    println!("\n✅ Example completed successfully!");
}
