use pesti_runner::CpuModel;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    let prompt = "The quick brown fox jumps over the lazy dog. ";
    let max_tokens = 20;

    println!("=== PESTI Autoregressive Generation (CPU) ===");
    println!("Model: {}", model_path);
    println!("Prompt: \"{}\"", prompt);
    println!("Max tokens: {}\n", max_tokens);

    let load_start = Instant::now();
    let (tokenizer_config, _tokenizer) =
        pesti_runner::load_tokenizer_from_gguf(Path::new(model_path))?;

    println!(
        "✓ Loaded tokenizer config in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );
    println!("  - Vocab size: {}", tokenizer_config.vocab_size);
    println!("  - BOS token: {:?}", tokenizer_config.bos_token_id);
    println!("  - EOS token: {:?}", tokenizer_config.eos_token_id);

    let cpu_model = CpuModel::load_gguf(Path::new(model_path))?;

    println!(
        "\n✓ Loaded model in {:.2}s",
        load_start.elapsed().as_secs_f32()
    );
    println!("  - Hidden size: {}", cpu_model.hidden_size);
    println!("  - Vocab size: {}", cpu_model.vocab_size);
    println!(
        "  - Token embeddings loaded: {}",
        cpu_model.token_embeddings.is_some()
    );
    println!(
        "  - Output weights loaded: {}",
        cpu_model.output_weights.is_some()
    );

    println!(
        "\n⚠️  NOTE: CpuModel only loads embeddings + output head for now."
    );
    println!("   For full transformer inference, use transformer_cpu::CpuTransformerModel");
    println!("   which loads all layer weights from GGUF.\n");

    let mut generated_tokens = Vec::new();
    let gen_start = Instant::now();

    println!("=== Generating {} tokens ===\n", max_tokens);

    let mut current_token: u32 = 100; // BOS token placeholder

    for step in 0..max_tokens {
        let hidden = cpu_model.embed(current_token, step)?;
        let logits = cpu_model.apply_output_head(&hidden)?;

        // Simple argmax sampling (no temperature yet)
        let next_token = argmax_sample(&logits);
        generated_tokens.push(next_token);

        if step == 0 {
            print!("Generated: ");
        }

        print!("{} ", next_token);

        current_token = next_token;
    }

    println!(
        "\n\n✓ Generation complete in {:.2}s",
        gen_start.elapsed().as_secs_f32()
    );
    println!(
        "  - Tokens/sec: {:.1}",
        max_tokens as f32 / gen_start.elapsed().as_secs_f32()
    );

    let output = format!(
        "Generated {} tokens from prompt: \"{}\"\nTokens: {:?}",
        generated_tokens.len(),
        prompt,
        generated_tokens
    );
    std::fs::write("generation_output.txt", &output)?;
    println!("  - Output saved to: generation_output.txt");

    println!(
        "\n⚠️  NOTE: This is a minimal implementation."
    );
    println!(
        "   The forward pass currently skips transformer layers (embed → output head only)."
    );
    println!(
        "   For full inference with attention and FFN, implement CpuTransformerModel loading."
    );

    Ok(())
}

fn argmax_sample(logits: &[f32]) -> u32 {
    if logits.is_empty() {
        return 0; // Fallback to first token
    }
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
