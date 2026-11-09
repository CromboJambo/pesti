//! Test with chat template applied (like pesti-server does)
use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use llama_cpp_2::model::LlamaChatMessage;

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .expect("Usage: test_chat <model.gguf>");

    println!("Loading model from: {}", model_path);
    let runner = LlamaRunner::builder(&model_path)
        .n_ctx(2048)
        .build()
        .expect("Failed to build runner");

    println!("Model loaded. Testing chat template...");

    // Build chat messages like pesti-server does
    let prompt = "What is 2+2?";
    let llama_messages: Vec<LlamaChatMessage> = vec![
        ("system", "You are a helpful assistant."),
        ("user", prompt),
    ]
    .iter()
    .filter_map(|(role, content)| {
        match LlamaChatMessage::new(role.to_string(), content.to_string()) {
            Ok(msg) => Some(msg),
            Err(e) => {
                eprintln!("Failed to create chat message: {}", e);
                None
            }
        }
    })
    .collect();

    println!("Built {} chat messages", llama_messages.len());

    match runner.apply_chat_template(&llama_messages, true) {
        Ok(tmpl) => {
            println!(
                "Chat template applied (first 100 chars): {}",
                tmpl.chars().take(100).collect::<String>()
            );

            let config = SamplingConfig {
                temperature: 0.7,
                top_k: 50,
                top_p: 0.9,
                max_tokens: 20,
                seed: 42,
                ..Default::default()
            };

            match runner.generate(&tmpl, &config) {
                Ok(result) => {
                    println!("Generated text: {}", result.text);
                    println!("Tokens generated: {}", result.generated_tokens);
                }
                Err(e) => {
                    eprintln!("Generation failed with chat template: {}", e);
                    std::process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("Chat template failed: {}", e);
            std::process::exit(1);
        }
    }
}
