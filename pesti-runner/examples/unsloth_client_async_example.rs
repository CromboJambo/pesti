//! Unsloth Studio Async SDK Example
//! 
//! Demonstrates high-throughput concurrent model inference using tokio + reqwest
//! 
//! This example shows:
//! - Concurrent model calls (3 models running in parallel)
//! - Stream-based responses for long-running generations
//! - Batch processing with async/await
//! 
//! Run with: cargo run --package pesti-runner --example unsloth_client_async_example

use pesti_runner::unsloth_client_async::{AsyncUnslothClient, ModelConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Unsloth Studio Async SDK Example ===\n");

    // Create async client (non-blocking)
    let client = AsyncUnslothClient::new("http://localhost:8888").await?;
    
    // Configure model parameters
    let config = ModelConfig::default();
    println!("Model Config:");
    println!("  Model: {}", config.model_name);
    println!("  Max Tokens: {}", config.max_tokens);
    println!("  Temperature: {}", config.temperature);
    println!();

    // Example 1: Concurrent model calls (3 models running in parallel)
    println!("--- Example 1: Concurrent Model Calls ---");
    let prompts = vec![
        "Explain Rust ownership in one sentence.",
        "What makes Rust memory-safe?",
        "How does async/await work in Rust?"
    ];

    // Run all 3 models concurrently (tokio::join!)
    let results = tokio::join!(
        client.run_model(&prompts[0], &config),
        client.run_model(&prompts[1], &config),
        client.run_model(&prompts[2], &config),
    );

    // Results from join! is a tuple, not an iterable
    let results_tuple = (results.0, results.1, results.2);
    
    for (i, result) in [
        (&results_tuple.0, 1usize),
        (&results_tuple.1, 2usize),
        (&results_tuple.2, 3usize),
    ].iter().enumerate() {
        match result.0 {
            Ok(chat_result) => {
                println!("Model {} response: {} tokens", result.1, chat_result.tokens_used);
            }
            Err(e) => {
                println!("Model {} error: {}", result.1, e);
            }
        }
    }
    println!("All models completed! (concurrent execution)\n");

    // Example 2: Stream-based response for long-running generation
    println!("--- Example 2: Streaming Response ---");
    match client.stream_model("Generate a poem about Rust...", &config).await {
        Ok(response) => {
            // Get content type from headers (async reqwest API)
            if let Some(content_type) = response.headers().get(reqwest::header::CONTENT_TYPE) {
                println!("✓ Got streaming response (content-type: {})", 
                         content_type.to_str().unwrap_or("unknown"));
            } else {
                println!("✓ Got streaming response");
            }
            // In real usage, you'd read the stream:
            // while let Some(chunk) = response.chunk().await? { ... }
        }
        Err(e) => {
            println!("Note: Streaming not available. Error: {}", e);
            println!("  (This is expected if Unsloth Studio isn't running)");
        }
    }
    println!();

    // Example 3: Batch processing with async/await
    println!("--- Example 3: Batch Processing ---");
    let batch_prompts = vec![
        "What is a closure in Rust?",
        "Explain lifetimes simply.",
        "What is the borrow checker?"
    ];

    // Process batch concurrently (tokio::spawn for true parallelism)
    let mut handles = Vec::new();
    for prompt in batch_prompts {
        let client_clone = client.clone(); // Clone async client
        let config_clone = config.clone();
        
        let handle = tokio::spawn(async move {
            client_clone.run_model(prompt, &config_clone).await
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    let mut processed = 0;
    for handle in handles {
        match handle.await {
            Ok(Ok(result)) => {
                println!("✓ Processed batch item {} ({} tokens)", processed + 1, result.tokens_used);
                processed += 1;
            }
            Ok(Err(e)) => {
                println!("✗ Batch item error: {}", e);
            }
            Err(e) => {
                println!("✗ Task panicked: {}", e);
            }
        }
    }
    println!("Batch completed: {} items processed\n", processed);

    // Example 4: Chat thread with multi-turn conversation
    println!("--- Example 4: Chat Thread ---");
    let messages = vec![
        pesti_runner::unsloth_client_async::ChatMessage {
            role: "user".to_string(),
            content: "What is Rust?".to_string()
        },
        pesti_runner::unsloth_client_async::ChatMessage {
            role: "assistant".to_string(),
            content: "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety.".to_string()
        },
        pesti_runner::unsloth_client_async::ChatMessage {
            role: "user".to_string(),
            content: "Why use it for LLMs?".to_string()
        }
    ];

    match client.chat(&messages, &config).await {
        Ok(chat_result) => {
            println!("✓ Chat response: {} tokens", chat_result.tokens_used);
        }
        Err(e) => {
            println!("Note: Chat API not available. Error: {}", e);
        }
    }

    println!("\n=== Async Example Complete ===");
    Ok(())
}
