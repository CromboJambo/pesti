//! Dump pesti's per-layer hidden states (all layers) + final logits for a
//! prompt, so we can diff each layer against the numpy reference
//! (`conformance-corpus/ref_forward.py`) and verify full-model conformance.
//!
//! This is the all-layer generalization of `dump_l0_intermediates.rs`. It runs
//! pesti's pure-Rust CPU forward path (per-layer `TransformerLayer::forward_with_cache`
//! — the SAME code the production decode loop uses) and mirrors `ref_forward.py`'s
//! output:
//!   - per-layer output norm + first-8 at the LAST prompt position
//!   - pre-head (final-norm) hidden norm + first-8
//!   - top-8 logits + argmax
//!
//! Usage:
//!   cargo run -p pesti-runner --release --example dump_all_layers -- <model.gguf> [tok1,tok2,...]
//!
//! With `--raw` it prints machine-parseable lines for automated diffing:
//!   cargo run -p pesti-runner --release --example dump_all_layers -- --raw <model.gguf> [toks]
//!
//! With `--dump DIR` it also writes the FULL per-layer vectors as raw f32 files
//! (`embed.f32`, `layer_<l>.f32`, `prehead.f32`, `logits.f32`) for a full-vector
//! diff against the numpy probe's saved arrays:
//!   cargo run -p pesti-runner --release --example dump_all_layers -- --dump /tmp/rust_probe <model.gguf>
//!
//! Default prompt is the same 10-token "fox" prompt ref_forward.py uses.
use std::path::Path;

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}
fn head8(v: &[f32]) -> String {
    v.iter()
        .take(8)
        .map(|x| format!("{x:.4}"))
        .collect::<Vec<_>>()
        .join(",")
}

const DEFAULT_PROMPT: &[u32] = &[785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut raw = false;
    let mut dump_dir: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--raw" => raw = true,
            "--dump" => match args.get(i + 1) {
                Some(d) => {
                    dump_dir = Some(d.clone());
                    i += 1;
                }
                None => {
                    eprintln!("error: --dump requires a directory argument");
                    std::process::exit(2);
                }
            },
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    if positional.is_empty() {
        eprintln!("usage: dump_all_layers [--raw] [--dump DIR] <model.gguf> [tok1,tok2,...]");
        std::process::exit(2);
    }
    let model_path = positional[0].clone();
    let toks: Vec<u32> = if positional.len() > 1 {
        positional[1]
            .split(',')
            .map(|s| s.parse::<u32>().expect("bad token id"))
            .collect()
    } else {
        DEFAULT_PROMPT.to_vec()
    };

    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let cfg = &model.config;
    eprintln!(
        "[CFG] arch={:?} n_layer={} n_head={} n_head_kv={} n_embd={} n_ffn={} rope_base={} rms_eps={:.3e}",
        cfg.arch,
        cfg.num_layers,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.embed_dim,
        cfg.intermediate_dim,
        cfg.rope_base,
        cfg.rms_norm_eps
    );

    // Run the prompt token-by-token through the CPU KV-cache path (mirrors
    // ref_forward.py's per-position loop). We capture per-layer outputs at the
    // LAST position only, to match the reference.
    let last = toks.len() - 1;
    let mut per_layer: Vec<Vec<f32>> = Vec::with_capacity(cfg.num_layers);
    let mut last_embed: Option<Vec<f32>> = None;

    for (pos, &tok) in toks.iter().enumerate() {
        let emb = model.embed(tok, pos)?;
        if pos == last {
            last_embed = Some(emb.clone());
            if raw {
                println!("embed norm={:.6}", norm(&emb));
                println!("embed head=[{}]", head8(&emb));
            } else {
                println!(
                    "[P] pos={} tok={} embed norm={:.4} head=[{}]",
                    pos,
                    tok,
                    norm(&emb),
                    head8(&emb)
                );
            }
        }

        // Initialize CPU KV caches on first use (one LayerKvCache per layer).
        let caches = model.cpu_kv_caches.get_or_insert_with(|| {
            model
                .layers
                .iter()
                .map(|l| {
                    pesti_runner::transformer::kv_cache::LayerKvCache::new(
                        l.attention.num_kv_heads,
                        l.attention.head_dim,
                        model.config.max_seq_len,
                    )
                })
                .collect()
        });

        if pos != last {
            // Not the last position: run all layers, discard intermediate states.
            let mut h = emb;
            for (l, layer) in model.layers.iter().enumerate() {
                h = layer.forward_with_cache(&h, &mut caches[l], pos);
            }
            continue;
        }

        // Last position: step layer-by-layer to capture each output.
        let mut h = emb;
        for (l, layer) in model.layers.iter().enumerate() {
            h = layer.forward_with_cache(&h, &mut caches[l], pos);
            per_layer.push(h.clone());
        }
    }

    // Final norm + logits (mirrors ref_forward.py: h = rmsnorm(last_hidden,
    // output_norm); logits = OUT @ h).
    let pre_head = match &model.final_norm {
        Some(n) => n.forward(&per_layer.last().unwrap(), 1),
        None => per_layer.last().unwrap().clone(),
    };
    let logits = model.apply_output_head(&pre_head)?;

    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top8: Vec<usize> = idx.into_iter().take(8).collect();

    // Optional: dump full vectors for a full-vector diff against the numpy probe.
    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir)?;
        let write = |name: &str, v: &[f32]| {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/{name}.f32"), &bytes)
        };
        if let Some(emb) = &last_embed {
            write("embed", emb)?;
        }
        for (l, out) in per_layer.iter().enumerate() {
            write(&format!("layer_{l}"), out)?;
        }
        write("prehead", &pre_head)?;
        write("logits", &logits)?;
        eprintln!(
            "[DUMP] wrote {} full vectors to {dir}/",
            per_layer.len() + 3
        );
    }

    if raw {
        for (l, out) in per_layer.iter().enumerate() {
            println!("layer {l} norm={:.6}", norm(out));
            println!("layer {l} head=[{}]", head8(out));
        }
        println!("prehead norm={:.6}", norm(&pre_head));
        println!("prehead head=[{}]", head8(&pre_head));
        println!(
            "top8 tokens=[{}]",
            top8.iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "top8 logits=[{}]",
            top8.iter()
                .map(|i| format!("{:.3}", logits[*i]))
                .collect::<Vec<_>>()
                .join(",")
        );
        println!("argmax={}", top8[0]);
    } else {
        for (l, out) in per_layer.iter().enumerate() {
            println!(
                "[P] layer={} pos={} norm={:.4} head=[{}]",
                l,
                last,
                norm(out),
                head8(out)
            );
        }
        println!(
            "[P] pre-head norm={:.4} head=[{}]",
            norm(&pre_head),
            head8(&pre_head)
        );
        println!(
            "\n=== PESTI (Rust CPU, {} layers) ===",
            model.config.num_layers
        );
        println!("top-8 tokens: {:?}", top8);
        println!(
            "top-8 logits: [{}]",
            top8.iter()
                .map(|i| format!("{:.3}", logits[*i]))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("argmax: {}", top8[0]);
    }

    Ok(())
}
