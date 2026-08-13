//! Integration test for loading external GGUF models (e.g., Bonsai 27B)
//! 
//! This test demonstrates how to manually register a model path before loading,
//! which is useful for models not in the standard discovery paths.

use pesti_runner::runtime::Runtime;
use pesti_runner::error::Result;
use pesti_runner::registry::{ModelEntry, ModelFormat};
use tokio;
use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber;

// Define model constants for the integration test
const BONSAI_MODEL_NAME: &str = "bonsai-27b";
// The absolute path to the Bonsai 27B GGUF file provided by the user.
const BONSAI_GGUF_PATH: &str = "/mnt/data/state/ai/lmstudio/models/lmstudio-community/Bonsai-27B-GGUF/Bonsai-27B-Q1_0.gguf";

/// Integration test to load and run inference on an external GGUF model.
#[tokio::test]
async fn test_bonsai_integration() -> Result<()> {
    // Set up tracing for visibility during the test run
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    println!("\n=============================================");
    println!("  Starting Bonsai 27B Integration Test...");
    println!("=============================================\n");

    // --- Manual Model Registration ---
    // The PESTI Runtime supports manual model registration for models outside
    // standard discovery paths. This simulates what filesystem discovery would do.
    
    let mut runtime = Runtime::new(); // Initialize with default config
    
    // Register the model manually using the public API
    println!("Registering model: {}", BONSAI_MODEL_NAME);
    runtime.register_model(ModelEntry {
        name: BONSAI_MODEL_NAME.to_string(),
        base_path: PathBuf::from(BONSAI_GGUF_PATH),
        lora_path: None,
        template: Some("chatml".to_string()), // Bonsai uses ChatML template
        ctx_len: Some(262144), // Using known Bonsai context window
        n_threads: Some(0), // Auto-detect threads
    });

    // Verify the model is registered
    let spec = runtime.model_spec(BONSAI_MODEL_NAME);
    assert!(spec.is_some(), "Model should be registered after manual registration");
    println!("✅ Model registered successfully");

    // Attempt to load model via the public API
    println!("\nAttempting to load Bonsai 27B...");
    match runtime.load_model(BONSAI_MODEL_NAME).await {
        Ok(_) => println!("✅ Model loaded successfully into runtime."),
        Err(e) => panic!("❌ Failed to load Bonsai 27B: {:?}", e),
    }

    // Verification Check
    let state = runtime.model_info().await;
    assert!(state.is_some(), "Runtime reports no model is currently active after attempted load.");
    println!("✅ Model state verified: {} loaded.", state.unwrap().name);


    // Run inference
    println!("\nRunning sample generation with Bonsai...");
    let prompt = "Who are you and what is your primary function?";
    let sampling_config = pesti_runner::llama::SamplingConfig {
        temperature: 0.1, // Low temp for deterministic test response
        top_k: 40,
        top_p: 0.9,
        seed: Some(123),
    };

    let result = runtime.generate(prompt, &sampling_config)?;

    // Final Assertion Check
    println!("Generation complete.");
    assert!(result.eval_ms > 0, "Inference time should be greater than zero.");
    assert!(!result.token_ids.is_empty(), "Generated token sequence must not be empty.");
    
    // Print the first few tokens for human inspection of output quality
    println!("First 5 tokens: {:?}", &result.token_ids[..std::cmp::min(5, result.token_ids.len())]);

    Ok(())
}

/// Alternative test using filesystem discovery (recommended approach)
#[tokio::test]
async fn test_bonsai_discovery() -> Result<()> {
    // This is the preferred way: place GGUF files in discovery paths and let
    // Runtime auto-discover them. No manual registration needed!
    
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    println!("\n=============================================");
    println!("  Testing Model Discovery (preferred approach)...");
    println!("=============================================\n");

    let runtime = Runtime::new();
    
    // List available models from discovery paths
    let available = runtime.list_available();
    println!("Available models: {:?}", available);
    
    // If bonsai is in the list, load it directly
    if available.contains(&BONSAI_MODEL_NAME.to_string()) {
        println!("✅ Bonsai found in discovered models");
        runtime.load_model(BONSAI_MODEL_NAME).await?;
        let state = runtime.model_info().await;
        assert!(state.is_some());
        println!("✅ Model loaded via discovery");
    } else {
        println!("ℹ️  Bonsai not found in discovery paths (expected if file not copied)");
        println!("   Copy to: $CRABJAR_MODEL_PATHS");
    }

    Ok(())
}
