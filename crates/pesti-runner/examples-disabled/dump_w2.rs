//! Dump the model's actual ffn_down (w2) weight matrix as f32 so we can
//! compare it against the byte-exact reference dequant.
//!
//! Usage: cargo run -p pesti-runner --release --features cuda
//!   --example dump_w2 -- <path> <out_prefix>
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dump_w2 <model.gguf> <out_prefix>");
        std::process::exit(2);
    }
    let model_path = args[1].clone();
    let out_prefix = args[2].clone();
    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    let model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let w2 = &model.layers[0].feed_forward.w2;
    eprintln!(
        "w2: in_features={} out_features={} weight.len={}",
        w2.in_features,
        w2.out_features,
        w2.weight.len()
    );
    let bytes: Vec<u8> = w2.weight.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(format!("{out_prefix}.f32"), &bytes)?;
    println!("{}", w2.weight.len());
    Ok(())
}
