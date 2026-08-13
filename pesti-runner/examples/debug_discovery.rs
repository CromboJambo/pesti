//! Debug test to see what files are being scanned

use std::path::PathBuf;
use pesti_runner::registry::{ModelDiscovery, ModelFormat};

fn main() {
    let model_dir = std::env::var("CRABJAR_MODEL_PATHS")
        .unwrap_or_else(|_| "/home/crombo/pesti-models".to_string());

    println!("=== Debug: Scanning {} ===\n", model_dir);

    let discovery = ModelDiscovery::new();
    
    // Add the path manually
    let mut debug_discovery = ModelDiscovery::new();
    debug_discovery.add_search_path(PathBuf::from(&model_dir));
    
    match debug_discovery.discover_models() {
        Ok(models) => {
            println!("Found {} models:", models.len());
            for model in &models {
                println!("  - {} ({:?})", model.name, model.format);
                if let Some(size) = model.size_bytes {
                    println!("    Size: {:.1} MB", size as f64 / 1024.0 / 1024.0);
                }
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }

    // Also check if the file exists
    let test_path = PathBuf::from(&model_dir).join("qwen2.5-0.5b-instruct-q4_k_m.gguf");
    println!("\nTest path exists: {}", test_path.exists());
    println!("File extension: {:?}", test_path.extension());
}
