//! Verify the Q5_0/Q6_K/Q8_0 dequant fixes by running pesti's CPU layer-0
//! forward on token 785 and comparing against the numpy reference
//! (conformance-corpus/probe_layer0.py, same token).
//!
//! Reference (q4_k_m, tok=785):
//!   embed norm      = 0.3682
//!   after-ffn (L0)  = 7.9472   head=[-0.1803,-0.0320,-0.1383,-0.4192,0.0638,-0.0576,0.0480,-0.2649]
//!
//! Usage: cargo run -p pesti-runner --release --example verify_layer0_dequant -- <path>

use std::path::Path;

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
fn head(v: &[f32]) -> String {
    v.iter().take(8).map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: verify_layer0_dequant <model.gguf>");
        std::process::exit(2);
    }
    let model_path = args[1].clone();
    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    let model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let tok: u32 = 785; // "The"
    let emb = model.embed(tok, 0)?;
    println!("[PESTI] tok={tok} embed norm={:.4} head=[{}]", norm(&emb), head(&emb));

    // CPU path, single token, pos 0. RoPE is identity at pos 0; attention over
    // one position gives attn_out = v, so this matches the numpy reference.
    let l0 = model.layers[0].forward(&emb, 1, 1, 0);
    println!("[PESTI] L0 after-ffn norm={:.4} head=[{}]", norm(&l0), head(&l0));

    // Ratio vs reference after-ffn (7.9472). The old bug was ~8x.
    let ref_after_ffn = 7.9472f32;
    println!("[PESTI] L0 norm ratio vs ref = {:.4}  (old bug was ~8.0)", norm(&l0) / ref_after_ffn);

    Ok(())
}
