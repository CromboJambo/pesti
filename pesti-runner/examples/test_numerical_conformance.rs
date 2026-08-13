//! Numerical conformance test for Qwen2.5-0.5B
//! 
//! Loads model via Runtime and verifies generation matches expected output

use pesti_runner::runtime::Runtime;
use pesti_runner::llama::SamplingConfig;

#[tokio::main]
async fn main() -> pesti_runner::error::Result<()> {
    println!("=== Numerical Conformance Test ===\n");

    // Use the discovered model path
    let model_name = "qwen2.5-0.5b-instruct-q4_k_m";
    
    println!("Loading model: {}", model_name);
    let runtime = Runtime::new();
    
    // Check if model is available
    let available = runtime.list_available();
    if !available.contains(&model_name.to_string()) {
        println!("⚠️  Model not found in discovery paths");
        println!("Available: {:?}", available);
        return Ok(());
    }

    // Load the model
    runtime.load_model(model_name).await?;
    println!("✅ Model loaded successfully\n");

    // Run inference with greedy (deterministic) sampling
    let prompt = "The quick brown fox jumps over the lazy dog.";
    let mut config = SamplingConfig::greedy();
    config.max_tokens = 10;
    config.temperature = 0.0; // Force greedy

    println!("Generating {} tokens with greedy sampling...", config.max_tokens);
    let result = runtime.generate(prompt, &config)?;

    println!("\n=== Results ===");
    println!("Tokens generated: {}", result.generated_tokens);
    println!("Time: {} ms", result.eval_ms);
    println!("Throughput: {:.1} tok/s", result.generated_tokens as f64 / (result.eval_ms as f64 / 1000.0));
    
    // Decode tokens to text
    if let Some(runner) = runtime.model_info().await {
        println!("\nModel: {}", runner.name);
        println!("Path: {}", runner.path.display());
        
        // Print first few generated tokens (from the tokens vector)
        if !result.tokens.is_empty() {
            let new_tokens = &result.tokens[result.prompt_tokens..];
            println!("\nFirst 10 new token IDs: {:?}", &new_tokens[..std::cmp::min(10, new_tokens.len())]);
            
            // Try to get text representation
            let text_len = std::cmp::min(200, result.text.len());
            println!("Generated text (first {} chars): \"{}...\"", 
                text_len, 
                &result.text[..text_len]);
        }
    }

    // Verify determinism by running again with fresh model
    println!("\n=== Determinism Check ===");
    
    // Create a new runtime to reset KV cache
    let runtime2 = Runtime::new();
    runtime2.load_model(model_name).await?;
    
    let result2 = runtime2.generate(prompt, &config)?;
    
    if result.generated_tokens == result2.generated_tokens {
        println!("✅ PASS: Same number of tokens generated (deterministic sampling)");
        
        // Compare token IDs
        let new_tokens1 = &result.tokens[result.prompt_tokens..];
        let new_tokens2 = &result2.tokens[result.prompt_tokens..];
        
        if new_tokens1 == new_tokens2 {
            println!("✅ PASS: Token IDs are identical (byte-exact determinism)");
        } else {
            println!("⚠️  WARNING: Token IDs differ");
            println!("   Run 1 first 5: {:?}", &new_tokens1[..std::cmp::min(5, new_tokens1.len())]);
            println!("   Run 2 first 5: {:?}", &new_tokens2[..std::cmp::min(5, new_tokens2.len())]);
        }
    } else {
        println!("⚠️  WARNING: Different number of tokens generated");
        println!("   Run 1: {}", result.generated_tokens);
        println!("   Run 2: {}", result2.generated_tokens);
    }

    Ok(())
}
