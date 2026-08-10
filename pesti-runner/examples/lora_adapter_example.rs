//! Example: Using LoRA adapters with PESTI

use pesti_runner::peft::{Adapter, AdapterConfig, LoRAAdapter};

fn main() {
    println!("=== PEFT LoRA Adapter Example ===\n");

    // 1. Create a LoRA adapter configuration
    let config = AdapterConfig::lora(8, 16.0);
    println!(
        "Configuration: rank={}, scaling={}",
        config.rank, config.scaling
    );

    // 2. Initialize adapter with random weights
    let in_features = 512;
    let out_features = 1024;
    let adapter = LoRAAdapter::new_random(in_features, out_features, &config).unwrap();

    println!("\nAdapter created:");
    println!("  - In features: {}", adapter.in_features);
    println!("  - Out features: {}", adapter.out_features);
    println!("  - Rank: {}", adapter.rank());
    println!("  - Scaling: {}", adapter.scaling());
    println!("  - Is initialized: {}", adapter.is_initialized());

    // 3. Forward pass through the adapter
    let batch_size = 2;
    let x: Vec<f32> = (0..batch_size * in_features)
        .map(|i| i as f32 * 0.01)
        .collect();

    println!("\nForward pass:");
    println!("  - Input shape: [{} x {}]", batch_size, in_features);

    match adapter.forward(&x, batch_size) {
        Ok(output) => {
            println!("  - Output shape: [{} x {}]", batch_size, out_features);
            println!("  - Output (first 5 values): {:?}", &output[..5.min(output.len())]);
        }
        Err(e) => {
            println!("  - Error: {}", e);
        }
    }

    // 4. Merge adapter into base weights (for deployment)
    let base_weights: Vec<f32> = (0..out_features * in_features)
        .map(|i| i as f32 * 0.1)
        .collect();

    println!("\nMerging adapter into base weights...");
    match adapter.merge_into(&base_weights, None) {
        Ok((_merged_weights, _merged_bias)) => {
            println!(
                "  - Merged weights shape: [{} x {}]",
                out_features, in_features
            );
        }
        Err(e) => {
            println!("  - Error: {}", e);
        }
    }

    // 5. Calculate parameter count
    let base_params = (out_features * in_features) as f64;
    let lora_params = (in_features * config.rank + config.rank * out_features) as f64;
    let param_ratio = (lora_params / base_params) * 100.0;

    println!("\n=== Parameter Efficiency ===");
    println!("Base model parameters: {:.2}M", base_params / 1_000_000.0);
    println!("LoRA adapter parameters: {:.4}M", lora_params / 1_000_000.0);
    println!("Parameter ratio: {:.2}%", param_ratio);

    println!("\n=== Example Complete ===");
}
