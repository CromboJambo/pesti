//! Simple test to verify model discovery works with environment variable

use pesti_runner::runtime::Runtime;

#[tokio::main]
async fn main() -> pesti_runner::error::Result<()> {
    println!("=== Testing Model Discovery ===\n");

    let runtime = Runtime::new();

    // List available models
    let available = runtime.list_available();
    println!("Discovered {} models:", available.len());

    for name in &available {
        println!("  - {}", name);
    }

    // Try to load the qwen2.5-0.5b model
    if available.contains(&"qwen2.5-0.5b-instruct-q4_k_m".to_string()) {
        println!("\n✓ Found qwen2.5-0.5b-instruct-q4_k_m");

        println!("Loading model...");
        runtime.load_model("qwen2.5-0.5b-instruct-q4_k_m").await?;
        println!("✅ Model loaded successfully!");

        let state = runtime.model_info().await;
        if let Some(s) = state {
            println!("Model: {}", s.name);
            println!("Path: {}", s.path.display());
            println!("Format: {:?}", s.format);
        }
    } else {
        println!("\n⚠️  qwen2.5-0.5b-instruct-q4_k_m not found");
        println!("Available models: {:?}", available);
    }

    Ok(())
}
