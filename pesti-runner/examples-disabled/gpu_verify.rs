//! Quick GPU forward pass verification test
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== GPU Forward Pass Verification ===\n");

    // Load weights
    let t0 = Instant::now();
    let _weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    println!("Weights loaded in {:.2}s", t0.elapsed().as_secs_f32());

    // Load model
    let t1 = Instant::now();
    let mut model = pesti_runner::transformer::LlamaModel::load_gguf(Path::new(model_path))?;
    println!("Model built in {:.2}s", t1.elapsed().as_secs_f32());

    // Check GPU availability
    if let Some(ref dispatch) = model.dispatch {
        println!("GPU available: {}", dispatch.gpu_available());
    } else {
        println!("No dispatch context (CPU-only mode)");
    }

    // Simple generation test - just 3 tokens
    let prompt = "Hello";
    println!("\nPrompt: \"{}\"", prompt);

    // Encode
    let tok = model.tokenizer.as_ref().ok_or("No tokenizer")?;
    let tokens = tok.encode(prompt)?;
    println!("Encoded to {} tokens", tokens.len());

    // Generate just 3 tokens
    let sampling_config = pesti_runner::transformer::SamplingConfig {
        temperature: 0.8,
        top_p: 0.95,
        top_k: 40,
        seed: Some(42),
    };

    let mut rng = StdRng::seed_from_u64(42);
    println!("\nGenerating 3 tokens...");

    let start = Instant::now();
    let generated = model.generate(&tokens, 3, &sampling_config, &mut rng, &[0])?;
    let elapsed = start.elapsed();

    let throughput = generated.len() as f64 / elapsed.as_secs_f64();
    println!(
        "Generated {} tokens in {:.3}s ({:.1} tok/s)",
        generated.len(),
        elapsed.as_secs_f32(),
        throughput
    );

    // Decode
    if let Some(ref tok) = model.tokenizer {
        if let Ok(text) = tok.decode(&generated) {
            println!(
                "\nGenerated text: \"{}\"",
                text.chars().take(200).collect::<String>()
            );
        }
    }

    println!("\n✅ Test complete!");

    Ok(())
}
