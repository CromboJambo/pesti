//! Dump pesti's layer-0 intermediates (using its public Linear/RmsNorm/Attention
//! objects) so we can compare each sub-op against the numpy reference (Method B).
//!
//! Usage: cargo run -p pesti-runner --release --features cuda
//!   --example dump_l0_intermediates -- <path>
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
        eprintln!("usage: dump_l0_intermediates <model.gguf>");
        std::process::exit(2);
    }
    let model_path = args[1].clone();
    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    let model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let tok: u32 = 785;
    let emb = model.embed(tok, 0)?;
    println!("[P] embed       norm={:.4} head=[{}]", norm(&emb), head(&emb));

    let l0 = &model.layers[0];
    let a = l0.attention_norm.forward(&emb, 1);
    println!("[P] attn_input  norm={:.4} head=[{}]", norm(&a), head(&a));

    let q = l0.attention.wq.forward(&a, 1);
    let k = l0.attention.wk.forward(&a, 1);
    let v = l0.attention.wv.forward(&a, 1);
    println!("[P] q           norm={:.4} head=[{}]", norm(&q), head(&q));
    println!("[P] k           norm={:.4} head=[{}]", norm(&k), head(&k));
    println!("[P] v           norm={:.4} head=[{}]", norm(&v), head(&v));

    // attention.forward ALREADY applies the wo projection (layer.rs), so its
    // output IS attn_proj. Do NOT apply wo again.
    let attn_proj = l0.attention.forward(&a, 1, 1, 0);
    println!("[P] attn_proj   norm={:.4} head=[{}]", norm(&attn_proj), head(&attn_proj));

    let h: Vec<f32> = (0..emb.len()).map(|i| emb[i] + attn_proj[i]).collect();
    println!("[P] h(after-attn) norm={:.4} head=[{}]", norm(&h), head(&h));

    let f = l0.ffn_norm.forward(&h, 1);
    println!("[P] ffn_input   norm={:.4} head=[{}]", norm(&f), head(&f));

    let gate = l0.feed_forward.w1.forward(&f, 1);
    let up = l0.feed_forward.w3.forward(&f, 1);
    println!("[P] gate        norm={:.4} head=[{}]", norm(&gate), head(&gate));
    println!("[P] up          norm={:.4} head=[{}]", norm(&up), head(&up));

    // silu(gate)*up
    let swiglu: Vec<f32> = (0..gate.len())
        .map(|i| {
            let s = if gate[i] >= 0.0 {
                1.0 / (1.0 + (-gate[i]).exp())
            } else {
                gate[i] / (1.0 + gate[i].exp())
            };
            s * gate[i] * up[i]
        })
        .collect();
    let down = l0.feed_forward.w2.forward(&swiglu, 1);
    println!("[P] down        norm={:.4} head=[{}]", norm(&down), head(&down));

    let out: Vec<f32> = (0..h.len()).map(|i| h[i] + down[i]).collect();
    println!("[P] L0 out      norm={:.4} head=[{}]", norm(&out), head(&out));

    Ok(())
}
