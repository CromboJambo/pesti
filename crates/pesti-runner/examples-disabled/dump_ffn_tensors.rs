//! Dump pesti's layer-0 FFN intermediates (gate, up, swiglu, down) to f32
//! files at full precision so we can diff each against the numpy reference
//! and localize whether the `down` divergence is in the swiglu input or in
//! the w2 GEMM itself.
//!
//! Usage: cargo run -p pesti-runner --release --features cuda
//!   --example dump_ffn_tensors -- <model.gguf> <out_prefix>
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: dump_ffn_tensors <model.gguf> <out_prefix>");
        std::process::exit(2);
    }
    let model_path = args[1].clone();
    let out_prefix = args[2].clone();
    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    let model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let tok: u32 = 785;
    let emb = model.embed(tok, 0)?;
    let l0 = &model.layers[0];

    let a = l0.attention_norm.forward(&emb, 1);
    let attn_proj = l0.attention.forward(&a, 1, 1, 0);
    let h: Vec<f32> = (0..emb.len()).map(|i| emb[i] + attn_proj[i]).collect();
    let f = l0.ffn_norm.forward(&h, 1);

    let gate = l0.feed_forward.w1.forward(&f, 1);
    let up = l0.feed_forward.w3.forward(&f, 1);
    let swiglu: Vec<f32> = (0..gate.len())
        .map(|i| {
            // Must match the FIXED library sigmoid (layer.rs swiglu):
            //   x >= 0 : 1/(1+e^{-x});  x < 0 : e^{x}/(1+e^{x})
            let s = if gate[i] >= 0.0 {
                1.0 / (1.0 + (-gate[i]).exp())
            } else {
                gate[i].exp() / (1.0 + gate[i].exp())
            };
            s * gate[i] * up[i]
        })
        .collect();
    let down = l0.feed_forward.w2.forward(&swiglu, 1);

    let dump = |name: &str, v: &[f32]| {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(format!("{out_prefix}.{name}.f32"), &bytes)?;
        eprintln!("{name}: len={}", v.len());
        Ok::<(), Box<dyn std::error::Error>>(())
    };
    dump("f", &f)?;
    dump("gate", &gate)?;
    dump("up", &up)?;
    dump("swiglu", &swiglu)?;
    dump("down", &down)?;
    Ok(())
}
