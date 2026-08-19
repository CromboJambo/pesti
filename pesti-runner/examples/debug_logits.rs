//! Debug: print PESTI's tokens + top-10 logits for a known prompt, to compare
//! against llama.cpp ground truth. This isolates whether the bug is in the
//! tokenizer, the forward pass, or the output head.
use pesti_runner::transformer::LlamaModel;
use std::time::Instant;

fn main() {
    let model_path =
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

    let t0 = Instant::now();
    let mut model =
        LlamaModel::load_gguf(std::path::Path::new(model_path)).expect("failed to load model");
    println!("[load] {} ms", t0.elapsed().as_millis());

    let prompt_text = "Once upon a time in the land of Rust,";
    let tokens = {
        let tokenizer = model.tokenizer.as_ref().expect("no tokenizer");
        tokenizer.encode(prompt_text).expect("failed to encode prompt")
    };
    println!("[tokens] {} tokens: {:?}", tokens.len(), tokens);
    println!("[ref ] [12522, 5193, 264, 882, 304, 279, 4268, 315, 33789, 11]");

    // Run prefill: feed each prompt token, keep the final logits
    let mut logits: Vec<f32> = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        let hidden = model.embed(tok, i).expect("embed failed");
        logits = model
            .forward_with_dispatch(&hidden, i)
            .expect("forward failed");
    }
    println!("[logits] {} values", logits.len());

    // Top-10 logits
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("[top10] PESTI:");
    for (rank, (id, val)) in indexed.iter().take(10).enumerate() {
        let tok = model
            .tokenizer
            .as_ref()
            .unwrap()
            .decode(&[*id as u32])
            .unwrap_or_default();
        println!(
            "  #{}  id={:<7}  logit={:+.4}  {:?}",
            rank + 1,
            id,
            val,
            tok
        );
    }

    // Reference: llama.cpp predicts " there" (token 279) as the first generated token
    println!("[ref ] llama.cpp first generated token: ' there' (id=279)");

    // Check where token 279 ranks in PESTI's logits
    let rank_279 = indexed.iter().position(|(id, _)| *id == 279);
    println!(
        "[check] token 279 ('there') ranks #{:?} in PESTI logits",
        rank_279.map(|r| r + 1)
    );
}
