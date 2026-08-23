//! Dump pesti's per-layer hidden states (all layers) + final logits for a
//! prompt, running the REAL GPU dispatch path (`forward_with_dispatch`), so we
//! can diff each layer against the numpy reference (`probe_all_layers.py`) and
//! verify GPU-path conformance — the direct continuation of Week 16's CPU
//! conformance (commit c476dae).
//!
//! This is the GPU counterpart of `dump_all_layers.rs` (which runs the CPU
//! path). It writes the EXACT file format `compare_full_vectors.py` expects:
//!   - embed.f32, layer_<l>.f32, prehead.f32, logits.f32  (raw LE f32)
//! and prints a per-layer norm + top-8 + argmax summary for eyeballing.
//!
//! How it works:
//!   - Prefills the prompt token-by-token through `forward_with_dispatch`
//!     (fills the GPU KV cache), capturing per-layer hidden states only at the
//!     LAST prompt position (via `model.capture_per_layer`).
//!   - `prehead` = final_norm(captured last layer) — RMSNorm is element-wise,
//!     so applying it on the CPU to the GPU's last-layer output is exact.
//!   - `logits` = the GPU logits `forward_with_dispatch` actually produced.
//!
//! Usage:
//!   cargo run -p pesti-runner --release --features cuda \
//!     --example dump_all_layers_gpu -- --dump /tmp/gpu_probe <model.gguf> [tok1,tok2,...]
//!
//! Default prompt is the same 10-token "fox" prompt the numpy oracle uses.
//! If no GPU is available the program ABORTS (it would otherwise silently
//! diff a CPU fallback, which is not what this tool is for).
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
    let mut dump_dir: Option<String> = None;
    let mut allow_cpu = false;
    let mut positional: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
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
            // Dry-run mode: allow the CPU fallback path so the dumper's
            // capture/dump logic can be validated without a free GPU.
            "--allow-cpu" => allow_cpu = true,
            other => positional.push(other.to_string()),
        }
        i += 1;
    }
    if positional.is_empty() {
        eprintln!("usage: dump_all_layers_gpu --dump DIR <model.gguf> [tok1,tok2,...]");
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

    let num_layers = {
        let c = &model.config;
        eprintln!(
            "[CFG] arch={:?} n_layer={} n_head={} n_head_kv={} n_embd={} n_ffn={} rope_base={} rms_eps={:.3e}",
            c.arch,
            c.num_layers,
            c.num_heads,
            c.num_kv_heads,
            c.embed_dim,
            c.intermediate_dim,
            c.rope_base,
            c.rms_norm_eps
        );
        c.num_layers
    };

    // Hard guard: this tool is for the GPU path. Abort loudly if the dispatch
    // context did not detect a GPU (it would otherwise silently run CPU).
    let ctx = model
        .dispatch
        .as_ref()
        .ok_or("dispatch context not initialized")?;
    if !ctx.gpu_available() {
        if !allow_cpu {
            eprintln!(
                "ERROR: GPU not available (dispatch.gpu_available()=false).\n\
                 This dumper runs the GPU dispatch path; refusing to fall back to CPU.\n\
                 Free VRAM / set CUDA_VISIBLE_DEVICES and retry (or pass --allow-cpu for a dry run)."
            );
            std::process::exit(3);
        }
        eprintln!("[CPU] WARNING: GPU not available; running CPU fallback (--allow-cpu dry run)");
    } else {
        eprintln!("[GPU] dispatch context reports GPU available: true");
    }

    let last = toks.len() - 1;

    // Prefill through the real GPU dispatch path. Capture per-layer states only
    // at the last position. The GPU KV cache is initialized on the first call
    // and persists across calls (as in production decode).
    let mut last_embed: Option<Vec<f32>> = None;
    let mut final_logits: Vec<f32> = Vec::new();
    for (pos, &tok) in toks.iter().enumerate() {
        let emb = model.embed(tok, pos)?;
        if pos == last {
            last_embed = Some(emb.clone());
            model.capture_per_layer = Some(Vec::new());
        }
        let logits = model.forward_with_dispatch(&emb, pos)?;
        if pos == last {
            final_logits = logits;
        }
    }

    let per_layer = model
        .capture_per_layer
        .take()
        .ok_or("per-layer capture was not populated")?;
    if per_layer.len() != num_layers {
        eprintln!(
            "ERROR: captured {} layers, expected {}",
            per_layer.len(),
            num_layers
        );
        std::process::exit(4);
    }

    // prehead = final_norm(last layer output). RMSNorm is element-wise, so the
    // CPU op applied to the GPU's last-layer output is exact.
    let pre_head = match &model.final_norm {
        Some(n) => n.forward(&per_layer[num_layers - 1], 1),
        None => per_layer[num_layers - 1].clone(),
    };

    // Summary for eyeballing (mirrors dump_all_layers.rs output).
    let mut idx: Vec<usize> = (0..final_logits.len()).collect();
    idx.sort_by(|&a, &b| final_logits[b].total_cmp(&final_logits[a]));
    let top8: Vec<usize> = idx.into_iter().take(8).collect();

    for (l, out) in per_layer.iter().enumerate() {
        eprintln!("[P] layer={l} norm={:.4} head=[{}]", norm(out), head8(out));
    }
    eprintln!(
        "[P] pre-head norm={:.4} head=[{}]",
        norm(&pre_head),
        head8(&pre_head)
    );
    eprintln!("[P] top-8 tokens: {:?}", top8);
    eprintln!(
        "[P] top-8 logits: [{}]",
        top8.iter()
            .map(|i| format!("{:.3}", final_logits[*i]))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!("[P] argmax: {}", top8[0]);

    // Report GPU fallback count: if >0, some ops silently fell back to CPU
    // (e.g. OOM on a busy GPU) and these are NOT pure-GPU numbers.
    let ctx = model.dispatch.as_ref().expect("dispatch context");
    let fb = ctx.gpu_fallback_count();
    if fb > 0 {
        eprintln!(
            "[P] WARNING: {fb} GPU op(s) fell back to CPU — results are a CPU/GPU mix, not pure-GPU"
        );
    } else if ctx.gpu_available() {
        eprintln!("[P] GPU fallback count: 0 (all ops ran on GPU)");
    }

    // Write raw LE f32 files in the exact format compare_full_vectors.py reads.
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
        write("logits", &final_logits)?;
        eprintln!(
            "[DUMP] wrote {} full vectors to {dir}/",
            per_layer.len() + 3
        );
    } else {
        eprintln!("[DUMP] (no --dump DIR given; nothing written to disk)");
    }

    Ok(())
}
