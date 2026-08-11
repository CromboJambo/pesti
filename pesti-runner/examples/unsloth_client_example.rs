//! Unsloth Studio Rust SDK Example
//!
//! Demonstrates how to use the type-safe UnslothClient to interact with
//! the Unsloth Studio API at http://localhost:8888
//!
//! This example shows:
//! - Creating a client with session cookie management
//! - Running model completions with configuration
//! - Multi-turn chat conversations
//! - Recipe workflow execution

use pesti_runner::unsloth_client::{
    ChatMessage, ModelConfig, Quantization, RecipeJob, UnslothClient,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unsloth Studio Rust SDK Example ===\n");

    // Create client (default 5-minute timeout for long generations)
    let client = UnslothClient::new("http://localhost:8888")?;

    // Configure model parameters
    let config = ModelConfig {
        model_name: "unsloth/llama-3-8b-Instruct-bnb-4bit".to_string(),
        max_tokens: 2048,
        temperature: 0.7,
        top_p: 0.9,
        quantization: Quantization::Bits4, // Memory-efficient quantization
    };

    println!("Model Config:");
    println!("  Model: {}", config.model_name);
    println!("  Max Tokens: {}", config.max_tokens);
    println!("  Temperature: {}", config.temperature);
    println!("  Top-P: {}", config.top_p);
    println!("  Quantization: {:?}", config.quantization);
    println!();

    // Example 1: Simple model completion
    println!("--- Example 1: Model Completion ---");
    let prompt = "What is Rust?";
    match client.run_model(prompt, &config) {
        Ok(result) => {
            println!("Response: {}", result.response);
            println!("Tokens Used: {}", result.tokens_used);
            println!("Duration: {:.2}ms", result.duration_ms);
        }
        Err(e) => {
            println!(
                "Note: API not available at {}. Error: {}",
                client.base_url(),
                e
            );
            println!("  (This is expected if Unsloth Studio isn't running)");
        }
    }
    println!();

    // Example 2: Multi-turn chat conversation
    println!("--- Example 2: Chat Conversation ---");
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful coding assistant.".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "What makes Rust memory-safe?".to_string(),
        },
    ];

    match client.chat(&messages, &config) {
        Ok(result) => {
            println!("Response: {}", result.response);
        }
        Err(e) => {
            println!("Note: API not available. Error: {}", e);
        }
    }
    println!();

    // Example 3: Recipe workflow execution (data processing)
    println!("--- Example 3: Data Recipe ---");
    use std::collections::HashMap;

    let mut config_map = HashMap::new();
    config_map.insert(
        "source_file".to_string(),
        serde_json::json!("/path/to/data.csv"),
    );
    config_map.insert(
        "transform".to_string(),
        serde_json::json!({
            "strip_newlines": true,
            "replace_semicolons": "|"
        }),
    );

    let recipe = RecipeJob {
        recipe_id: "warranty_data_cleaning".to_string(),
        config: config_map,
        seed_data: None,
    };

    match client.execute_recipe(&recipe) {
        Ok(result) => {
            println!("Recipe Status: {}", result.status);
            println!("Records Processed: {}", result.records_processed);
            println!("Duration: {:.2}ms", result.duration_ms);
            if let Some(output_path) = result.output_path {
                println!("Output: {}", output_path);
            }
        }
        Err(e) => {
            println!("Note: Recipe API not available. Error: {}", e);
        }
    }
    println!();

    // Example 4: Export recipe for visualization
    println!("--- Example 4: Export Recipe Definition ---");
    match client.export_recipe("warranty_data_cleaning") {
        Ok(recipe_json) => {
            println!("Recipe exported as JSON (for visualization tools):");
            println!("{}", serde_json::to_string_pretty(&recipe_json).unwrap());
        }
        Err(e) => {
            println!("Note: Export API not available. Error: {}", e);
        }
    }

    Ok(())
}
