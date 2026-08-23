//! Week 12 conformance probe: prefill-only forward pass with per-layer hidden
//! state dumps. Compare `[HIDDEN]` lines against `conformance-corpus/ref_forward.py`
//! (numpy reference, same fox prompt, pos=9 prefill step).
//!
//! Usage: PESTI_DEBUG_HIDDEN=1 PESTI_KV_MAX_SEQ=128 cargo run --release
//!   -p pesti-runner --features cuda --example week12_conformance_probe
//!
//! Reference (numpy, q4_k_m, fox prompt, last prompt position):
//!   layer=0  norm=3.8731
//!   layer=1  norm=7.1696
//!   layer=23 norm=50.5135
//!   pre-head norm=298.7678
//!   top-8 tokens: [220, 1416, 3555, 2585, 1096, 576, 758, 715]
//!   top-8 logits: [16.886, 16.697, 16.139, 16.094, 15.969, 15.945, 15.878, 15.484]
//!   argmax: 220 ("What")

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    println!("=== Week 12: Conformance Probe (prefill only) ===");
    println!("Model: {}", model_path);

    let weights = pesti_runner::load_gguf_weights(Path::new(model_path))?;
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;
    println!("Built model");

    let backend = pesti_runner::transformer::TokenizerBackend::MistralRs;
    let (_, tokenizer) =
        pesti_runner::transformer::load_tokenizer_from_gguf(Path::new(model_path), backend)?;

    let prompt = "The quick brown fox jumps over the lazy dog.";
    let prompt_tokens = tokenizer.encode(prompt)?;
    println!("Prompt tokens ({}): {:?}", prompt_tokens.len(), prompt_tokens);

    let mut logits = Vec::new();
    for (i, &tok) in prompt_tokens.iter().enumerate() {
        let hidden = model.embed(tok, i)?;
        logits = model.forward_with_dispatch(&hidden, i)?;
    }

    // Top-8 ranking of the final logits
    let mut ranked: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top8: Vec<(usize, f32)> = ranked.into_iter().take(8).collect();
    println!("[PROBE] top-8 tokens: {:?}", top8.iter().map(|(i, _)| i).collect::<Vec<_>>());
    println!("[PROBE] top-8 logits: {:?}", top8.iter().map(|(_, v)| v).collect::<Vec<_>>());
    println!("[PROBE] argmax: {}", top8[0].0);

    Ok(())
}
