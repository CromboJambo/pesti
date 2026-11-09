//! GPU tok/s benchmark test
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== GPU tok/s Benchmark ===\n");

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

    // Test prompt
    let prompt = "The quick brown fox";
    println!("\nPrompt: \"{}\"", prompt);

    // Encode
    let tok = model.tokenizer.as_ref().ok_or("No tokenizer")?;
    let tokens = tok.encode(prompt)?;
    println!("Encoded to {} tokens", tokens.len());

    // Generate 10 tokens for benchmark
    let sampling_config = pesti_runner::transformer::SamplingConfig {
        temperature: 0.7,
        top_p: 0.9,
        top_k: 40,
        seed: Some(42),
    };

    let mut rng = StdRng::seed_from_u64(42);
    println!("\nGenerating 10 tokens...");

    let start = Instant::now();
    let generated = model.generate(&tokens, 10, &sampling_config, &mut rng, &[0])?;
    let elapsed = start.elapsed();

    // Exclude first token (warmup) from throughput calculation
    let num_tokens = generated.len() - 1;
    let throughput = if num_tokens > 0 {
        num_tokens as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!(
        "Generated {} tokens in {:.3}s",
        generated.len(),
        elapsed.as_secs_f32()
    );
    println!("Throughput: {:.1} tok/s", throughput);

    // Decode
    if let Some(ref tok) = model.tokenizer {
        if let Ok(text) = tok.decode(&generated) {
            println!(
                "\nGenerated text: \"{}\"",
                text.chars().take(200).collect::<String>()
            );
        }
    }

    println!("\n✅ Benchmark complete!");

    Ok(())
}
