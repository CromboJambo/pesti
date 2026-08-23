//! Quick test to verify model discovery works with CRABJAR_MODEL_PATHS

use pesti_runner::runtime::Runtime;

#[tokio::main]
async fn main() -> pesti_runner::error::Result<()> {
    println!("=== Testing Model Discovery ===\n");

    let runtime = Runtime::new();

    // List available models from discovery paths
    let available = runtime.list_available();
    println!("Available models: {} total", available.len());

    if available.is_empty() {
        println!("\nℹ️  No models found in discovery paths");
        println!("   Set CRABJAR_MODEL_PATHS environment variable and copy GGUF files there");
        return Ok(());
    }

    println!("\nDiscovered models:");
    for name in &available {
        println!("  - {}", name);
    }

    Ok(())
}
