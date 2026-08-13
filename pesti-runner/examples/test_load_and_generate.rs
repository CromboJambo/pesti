//! Quick test to load and generate with a discovered model

use pesti_runner::runtime::Runtime;

#[tokio::main]
async fn main() -> pesti_runner::error::Result<()> {
    println!("=== Testing Model Loading & Generation ===\n");

    let runtime = Runtime::new();
    
    // List available models
    let available = runtime.list_available();
    println!("Available models: {} total", available.len());
    
    if available.is_empty() {
        println!("No models found!");
        return Ok(());
    }

    // Pick the first model (should be our qwen2.5-0.5b)
    let model_name = &available[0];
    println!("\nLoading model: {}", model_name);
    
    runtime.load_model(model_name).await?;
    println!("✅ Model loaded successfully!");

    // Get model info
    let state = runtime.model_info().await;
    if let Some(s) = state {
        println!("Model path: {}", s.path.display());
    }

    // Run inference
    println!("\nRunning inference...");
    let prompt = "The quick brown fox";
    let config = pesti_runner::llama::SamplingConfig {
        temperature: 0.7,
        top_k: 40,
        top_p: 0.9,
        seed: Some(42),
    };

    match runtime.generate(prompt, &config) {
        Ok(result) => {
            println!("✅ Generation complete!");
            println!("Tokens generated: {}", result.token_ids.len());
            println!("Time: {} ms", result.eval_ms);
            
            // Decode first few tokens
            if let Some(runner) = runtime.model_info().await {
                println!("Model: {}", runner.name);
            }
        }
        Err(e) => {
            println!("❌ Generation failed: {:?}", e);
        }
    }

    Ok(())
}
