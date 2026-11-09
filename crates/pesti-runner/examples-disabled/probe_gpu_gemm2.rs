//! Decisive probe: run the EXACT output-head GEMM that `forward_with_dispatch`
//! uses (A = real pre-head hidden state, B = real transposed output weights)
//! through BOTH the GPU path (`dispatch_gemm`) and the CPU path
//! (`dispatch_gemm_cpu`), and compare. If GPU returns zeros but CPU returns
//! real logits, the bug is in the GPU kernel for this specific input — not in
//! the model wiring.
//!
//! Build+run:
//!   cargo run -p pesti-runner --release --features cuda --example probe_gpu_gemm2 -- \
//!     conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf
use std::path::Path;

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf".into());

    let weights = pesti_runner::load_gguf_weights(Path::new(&model_path))?;
    let mut model = pesti_runner::transformer::LlamaModel::from_gguf_weights(weights)?;

    let toks: Vec<u32> = vec![785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13];
    let last = toks.len() - 1;

    // Run the full prompt through the dispatch path, capturing the last layer.
    for (pos, &tok) in toks.iter().enumerate() {
        let emb = model.embed(tok, pos)?;
        if pos == last {
            model.capture_per_layer = Some(Vec::new());
        }
        let _ = model.forward_with_dispatch(&emb, pos)?;
    }
    let per_layer = model.capture_per_layer.take().ok_or("no capture")?;
    let n_layers = per_layer.len();
    let pre_head = match &model.final_norm {
        Some(n) => n.forward(&per_layer[n_layers - 1], 1),
        None => per_layer[n_layers - 1].clone(),
    };
    eprintln!(
        "[probe] pre-head norm={:.4} (expect ~298.78)",
        norm(&pre_head)
    );

    // Build the EXACT A and B that forward_with_dispatch uses.
    let output = model.output.as_ref().ok_or("no output layer")?;
    let vocab = model.vocab_size as usize;
    let hidden = model.config.embed_dim as usize;
    let weight = &output.weight;
    let output_f16: Vec<half::f16> = (0..hidden)
        .flat_map(|k| (0..vocab).map(move |v| half::f16::from_f32(weight[v * hidden + k])))
        .collect();
    let a_f16: Vec<half::f16> = pre_head.iter().map(|&v| half::f16::from_f32(v)).collect();

    let ctx = model.dispatch.as_ref().ok_or("no dispatch")?;
    eprintln!(
        "[probe] gpu_available={} prefer_gpu={} fallback_before={}",
        ctx.gpu_available(),
        ctx.prefer_gpu(),
        ctx.gpu_fallback_count()
    );

    // GPU path.
    let fb0 = ctx.gpu_fallback_count();
    let gpu = ctx.dispatch_gemm(&a_f16, &output_f16, None, 1, vocab, hidden, 1.0, 0.0)?;
    let fb1 = ctx.gpu_fallback_count();
    // CPU path (same inputs).
    let cpu = ctx.dispatch_gemm_cpu(&a_f16, &output_f16, None, 1, vocab, hidden, 1.0, 0.0)?;

    eprintln!(
        "[probe] GPU logits: norm={:.4} first8={:?}",
        norm(&gpu),
        &gpu[..8.min(gpu.len())]
    );
    eprintln!(
        "[probe] CPU logits: norm={:.4} first8={:?}",
        norm(&cpu),
        &cpu[..8.min(cpu.len())]
    );
    eprintln!(
        "[probe] GPU fallback delta = {} (0 = ran on GPU, >0 = fell back to CPU)",
        fb1 - fb0
    );

    // Max abs diff GPU vs CPU.
    let mut maxdiff = 0.0f32;
    for (g, c) in gpu.iter().zip(cpu.iter()) {
        maxdiff = maxdiff.max((g - c).abs());
    }
    eprintln!("[probe] max|GPU-CPU| = {:.4}", maxdiff);

    // argmax for each.
    let amax = |v: &[f32]| -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    };
    eprintln!(
        "[probe] GPU argmax={}  CPU argmax={}  (expect 220)",
        amax(&gpu),
        amax(&cpu)
    );

    if norm(&gpu) < 1e-6 && norm(&cpu) > 1.0 {
        eprintln!(
            "\n==> BUG CONFIRMED: GPU GEMM returns zeros, CPU returns real logits for the SAME inputs."
        );
        eprintln!(
            "    The GPU kernel is broken for this input (or silently failing without an error)."
        );
    } else if (norm(&gpu) - norm(&cpu)).abs() < 1e-2 {
        eprintln!("\n==> GPU and CPU agree — the output-head GEMM is NOT the zero source.");
    } else {
        eprintln!("\n==> GPU and CPU DISAGREE (but GPU is not all-zero) — inspect the diff.");
    }
    Ok(())
}
