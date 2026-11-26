//! PESTI CLI — drop-in replacement for llama-cli
//!
//! Usage: pesti --model <MODEL> --prompt "Hello" [options]

use anyhow::Result;
use clap::{Parser, ValueHint};
use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "pesti", about = "PESTI command-line inference engine")]
struct Args {
    /// Model path to load (GGUF)
    #[arg(short = 'm', long = "model", value_hint = ValueHint::FilePath, required = true)]
    model: String,

    /// Prompt text (or use -f for file input)
    #[arg(short = 'p', long = "prompt")]
    prompt: Option<String>,

    /// File containing the prompt
    #[arg(short = 'f', long = "file", value_hint = ValueHint::FilePath)]
    file: Option<String>,

    /// Number of tokens to generate (default: 256, -1 for infinite)
    #[arg(short = 'n', long = "n-predict", default_value = "256")]
    n_predict: i32,

    /// Random seed (-1 for random)
    #[arg(short = 's', long = "seed", default_value = "42")]
    seed: u32,

    /// Temperature (0.0 = greedy, higher = more creative)
    #[arg(long = "temp", default_value = "0.7")]
    temperature: f64,

    /// Top-k sampling (0 = disabled)
    #[arg(long = "top-k", default_value = "40")]
    top_k: i32,

    /// Nucleus/top-p sampling (1.0 = disabled)
    #[arg(long = "top-p", default_value = "0.95")]
    top_p: f64,

    /// Repetition penalty (1.0 = disabled)
    #[arg(long = "repeat-penalty", default_value = "1.0")]
    repeat_penalty: f64,

    /// Number of recent tokens to consider for repetition penalty
    #[arg(long = "repeat-last-n", default_value = "64")]
    repeat_last_n: i32,

    /// CPU threads (0 = auto)
    #[arg(short = 't', long = "threads", default_value = "0")]
    threads: i32,

    /// Context window size
    #[arg(long = "ctx-size", default_value = "4096")]
    ctx_size: u32,

    /// Number of layers to offload to GPU (-1 = auto)
    #[arg(long = "gpu-layers", default_value = "-1")]
    gpu_layers: i32,

    /// Display prompt before generating (default: true)
    #[arg(long = "display-prompt", action = clap::ArgAction::SetTrue)]
    display_prompt: bool,

    /// Don't display prompt before generating
    #[arg(long = "no-display-prompt", action = clap::ArgAction::SetTrue)]
    no_display_prompt: bool,

    /// Show timings after generation (default: true)
    #[arg(long = "show-timings", action = clap::ArgAction::SetTrue)]
    show_timings: bool,

    /// Don't show timings after generation
    #[arg(long = "no-show-timings", action = clap::ArgAction::SetTrue)]
    no_show_timings: bool,

    /// Enable Q4_K KV cache for reduced memory and faster attention
    #[arg(long = "q4k-kv-cache", action = clap::ArgAction::SetTrue)]
    q4k_kv_cache: bool,

    /// Run attention spectral diagnostics (head health analysis) instead of generation.
    /// Accepts an optional second model path for quantization comparison:
    ///   --diagnose                     single-model diagnostics
    ///   --diagnose /path/to/model.gguf compare against this model
    #[arg(long = "diagnose", num_args = 0..=1, default_missing_value = "", require_equals = true)]
    diagnose: Option<String>,
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Determine if we should display prompt
    let display_prompt = args.display_prompt || !args.no_display_prompt;

    // Determine if we should show timings
    let show_timings = args.show_timings || !args.no_show_timings;

    // Get the actual seed (0 or -1 means random)
    let seed = if args.seed == 0 {
        // Use system time as entropy source for random seed
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        now.as_secs() as u32 ^ (now.subsec_nanos() as u32 >> 16)
    } else {
        args.seed
    };

    // Load prompt from file or use provided text
    let prompt_text = match (&args.prompt, &args.file) {
        (Some(p), _) => p.clone(),
        (None, Some(f)) => std::fs::read_to_string(f)?,
        (None, None) => {
            // Read from stdin
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            input.trim_end().to_string()
        }
    };

    if display_prompt {
        eprintln!("Prompt: {}", prompt_text);
    }

    // Build the runner
    let t_load = Instant::now();

    // Diagnostic mode: use native transformer with spectral profiler
    if let Some(compare_model) = &args.diagnose {
        if compare_model.is_empty() {
            // Single-model diagnostics
            run_diagnostics(&args, &prompt_text)?;
        } else {
            // Comparison mode: diagnose both models and compare
            run_diagnostics_compare(&args, &prompt_text, compare_model)?;
        }
        return Ok(());
    }

    let runner = LlamaRunner::builder(&args.model)
        .n_ctx(args.ctx_size)
        .n_threads(args.threads)
        .seed(seed)
        .n_gpu_layers(args.gpu_layers)
        .build()?;

    // Enable Q4_K KV cache if requested
    if args.q4k_kv_cache {
        runner.enable_q4k_kvcache();
        eprintln!("Q4_K KV cache enabled");
    }
    let load_time_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    // Model info
    let info = runner.model_info();
    eprintln!(
        "Model loaded: {} params, {} layers, {} context",
        info.n_params, info.n_layer, args.ctx_size
    );

    // Build sampling config
    let n_predict = if args.n_predict < 0 {
        u32::MAX
    } else {
        args.n_predict as u32
    };

    let sampling = SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repeat_penalty,
        repeat_last_n: args.repeat_last_n,
        max_tokens: n_predict,
        seed,
        ..Default::default()
    };

    // Generate
    let t_gen = Instant::now();
    let result = runner.generate(&prompt_text, &sampling)?;
    let gen_time_ms = t_gen.elapsed().as_secs_f64() * 1000.0;

    // Output generated text
    print!("{}", result.text);

    // Show timings if requested
    if show_timings {
        eprintln!("\n--- Timings ---");
        eprintln!("Model load: {:.2}ms", load_time_ms);
        eprintln!(
            "Prompt eval: {:.2}ms ({:.2} tok/s)",
            result.prompt_eval_ms,
            if result.prompt_eval_ms > 0.0 {
                result.prompt_tokens as f64 / (result.prompt_eval_ms / 1000.0)
            } else {
                0.0
            }
        );
        eprintln!(
            "Token eval: {:.2}ms ({:.2} tok/s)",
            result.eval_ms,
            if result.eval_ms > 0.0 {
                result.generated_tokens as f64 / (result.eval_ms / 1000.0)
            } else {
                0.0
            }
        );
        eprintln!("Total: {:.2}ms", gen_time_ms);
    }

    Ok(())
}

/// Load a model and run spectral diagnostics, returning the report.
fn load_and_diagnose(model_path: &str, prompt_text: &str) -> Result<Option<pesti_runner::diagnostics::spectral::SpectralProfileReport>> {
    use pesti_runner::transformer::LlamaModel;
    use pesti_runner::diagnostics::spectral::SpectralProfileReport;

    eprintln!("Loading model for diagnostics: {}", model_path);
    let t_load = Instant::now();
    let mut model = LlamaModel::load_gguf(std::path::Path::new(model_path))?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    eprintln!("  Loaded in {:.2}ms: {} layers, {} heads/layer", load_ms, model.config.num_layers, model.config.num_heads);

    let tokens = match &model.tokenizer {
        Some(tz) => tz.encode(prompt_text).map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?,
        None => {
            eprintln!("  ERROR: No tokenizer available.");
            return Ok(None);
        }
    };

    let seq_len = tokens.len();
    if seq_len == 0 {
        eprintln!("  ERROR: Prompt tokenized to 0 tokens.");
        return Ok(None);
    }
    eprintln!("  Tokenized to {} tokens", seq_len);

    let embeddings = match &model.token_embeddings {
        Some(emb) => emb.forward(&tokens.iter().map(|&t| t as f32).collect::<Vec<f32>>(), 1),
        None => {
            eprintln!("  ERROR: No token embeddings.");
            return Ok(None);
        }
    };

    eprintln!("  Running spectral analysis...");
    let t_diag = Instant::now();
    let report = model.diagnose_attention(&embeddings, seq_len)?;
    let diag_ms = t_diag.elapsed().as_secs_f64() * 1000.0;
    eprintln!("  Analysis complete in {:.2}ms", diag_ms);

    Ok(report)
}

/// Compare spectral diagnostics between two models (e.g., full-precision vs quantized).
fn run_diagnostics_compare(args: &Args, prompt_text: &str, compare_model_path: &str) -> Result<()> {
    use pesti_runner::diagnostics::spectral::{SpectralProfileReport, ProfileSummary};

    eprintln!("=== Quantization Comparison Mode ===");
    eprintln!("Model A (primary):   {}", args.model);
    eprintln!("Model B (compare):   {}", compare_model_path);
    eprintln!();

    let report_a = load_and_diagnose(&args.model, prompt_text)?;
    let report_b = load_and_diagnose(compare_model_path, prompt_text)?;

    match (report_a, report_b) {
        (Some(a), Some(b)) => {
            print_comparison_report(&a, &b);
        }
        (Some(_), None) => {
            eprintln!("Model B diagnostics failed. Cannot compare.");
        }
        (None, Some(_)) => {
            eprintln!("Model A diagnostics failed. Cannot compare.");
        }
        (None, None) => {
            eprintln!("Both models failed diagnostics. Cannot compare.");
        }
    }

    Ok(())
}

/// Print a side-by-side comparison of two spectral reports.
fn print_comparison_report(a: &pesti_runner::diagnostics::spectral::SpectralProfileReport, b: &pesti_runner::diagnostics::spectral::SpectralProfileReport) {
    eprintln!("\n=== Spectral Comparison Report ===");
    eprintln!("Model A: {}", a.model);
    eprintln!("Model B: {}", b.model);
    eprintln!();

    // Summary comparison
    eprintln!("{:<25} {:>12} {:>12} {:>12}", "Metric", "Model A", "Model B", "Δ (B-A)");
    eprintln!("{}", "-".repeat(65));

    let heads_a = a.summary.total_heads;
    let heads_b = b.summary.total_heads;
    eprintln!("{:<25} {:>12} {:>12} {:>12}", "Heads profiled", heads_a, heads_b, heads_b as i64 - heads_a as i64);

    let deg_a = a.summary.degenerate_heads;
    let deg_b = b.summary.degenerate_heads;
    eprintln!("{:<25} {:>12} {:>12} {:>12}", "Degenerate heads", deg_a, deg_b, deg_b as i64 - deg_a as i64);

    let deg_rate_a = if heads_a > 0 { a.summary.degenerate_heads as f64 / heads_a as f64 * 100.0 } else { 0.0 };
    let deg_rate_b = if heads_b > 0 { b.summary.degenerate_heads as f64 / heads_b as f64 * 100.0 } else { 0.0 };
    eprintln!("{:<25} {:>11.1}%  {:>11.1}%  {:>+11.1}%", "Degeneracy rate", deg_rate_a, deg_rate_b, deg_rate_b - deg_rate_a);

    let avg_rank_a = a.summary.avg_effective_rank;
    let avg_rank_b = b.summary.avg_effective_rank;
    eprintln!("{:<25} {:>12.2} {:>12.2} {:>+12.2}", "Avg effective rank", avg_rank_a, avg_rank_b, avg_rank_b - avg_rank_a);

    let min_rank_a = a.summary.min_effective_rank;
    let min_rank_b = b.summary.min_effective_rank;
    eprintln!("{:<25} {:>12.2} {:>12.2} {:>+12.2}", "Min effective rank", min_rank_a, min_rank_b, min_rank_b - min_rank_a);

    let max_deg_a = a.summary.max_degeneracy;
    let max_deg_b = b.summary.max_degeneracy;
    eprintln!("{:<25} {:>12.3} {:>12.3} {:>+12.3}", "Max degeneracy", max_deg_a, max_deg_b, max_deg_b - max_deg_a);

    // Verdict
    eprintln!();
    let rank_delta = avg_rank_b - avg_rank_a;
    let deg_delta = deg_rate_b - deg_rate_a;

    if deg_delta > 5.0 {
        eprintln!("⚠️  WARNING: Model B shows {:.1}% more degenerate heads than Model A.", deg_delta);
        eprintln!("   Quantization may be degrading attention quality significantly.");
    } else if deg_delta > 2.0 {
        eprintln!("ℹ️  NOTICE: Model B shows {:.1}% more degenerate heads — minor degradation.", deg_delta);
    } else if deg_delta < -2.0 {
        eprintln!("✅ Model B shows fewer degenerate heads than Model A (improvement).", );
    } else {
        eprintln!("✅ Spectral properties are comparable between models.");
    }

    if rank_delta.abs() > 1.0 {
        eprintln!("   Effective rank delta: {:+.2} — {}", rank_delta,
            if rank_delta < 0.0 { "Model B uses fewer attention dimensions" } else { "Model B uses more attention dimensions" });
    }

    // Per-layer comparison for significantly different layers
    eprintln!();
    eprintln!("=== Layer-by-Layer Degeneracy (only layers with Δ > 1 degenerate head) ===");
    let mut current_layer_a: Option<usize> = None;
    let mut layer_diffs: Vec<(usize, usize, usize)> = Vec::new(); // (layer, deg_a, deg_b)

    for head in &a.heads {
        if Some(head.layer) != current_layer_a {
            current_layer_a = Some(head.layer);
            let deg_in_layer_a = a.heads.iter().filter(|h| h.layer == head.layer && h.degeneracy_score >= 0.3).count();
            let deg_in_layer_b = b.heads.iter().filter(|h| h.layer == head.layer && h.degeneracy_score >= 0.3).count();
            if (deg_in_layer_a as i64 - deg_in_layer_b as i64).abs() > 1 {
                layer_diffs.push((head.layer, deg_in_layer_a, deg_in_layer_b));
            }
        }
    }

    if layer_diffs.is_empty() {
        eprintln!("  No layers with significant differences.");
    } else {
        for (layer, deg_a_count, deg_b_count) in &layer_diffs {
            let delta = *deg_b_count as i64 - *deg_a_count as i64;
            eprintln!("  Layer {:>2}: {} → {} ({:+})", layer, deg_a_count, deg_b_count, delta);
        }
    }
}

/// Run attention spectral diagnostics using the native transformer.
fn run_diagnostics(args: &Args, prompt_text: &str) -> Result<()> {
    use pesti_runner::transformer::LlamaModel;

    eprintln!("Loading model for spectral diagnostics...");
    let t_load = Instant::now();
    let mut model = LlamaModel::load_gguf(std::path::Path::new(&args.model))?;
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    eprintln!(
        "Model loaded in {:.2}ms: {} layers, {} heads/layer, embed_dim={}",
        load_ms,
        model.config.num_layers,
        model.config.num_heads,
        model.config.embed_dim,
    );

    // Tokenize the prompt
    let tokens = match &model.tokenizer {
        Some(tz) => tz.encode(prompt_text).map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?,
        None => {
            eprintln!("ERROR: Model has no tokenizer. Cannot run diagnostics without tokenization.");
            std::process::exit(1);
        }
    };

    let seq_len = tokens.len();
    if seq_len == 0 {
        eprintln!("ERROR: Prompt tokenized to 0 tokens.");
        std::process::exit(1);
    }
    eprintln!("Prompt tokenized to {} tokens", seq_len);

    // Get input embeddings from the model's embedding layer
    let embeddings = match &model.token_embeddings {
        Some(emb) => emb.forward(&tokens.iter().map(|&t| t as f32).collect::<Vec<f32>>(), 1),
        None => {
            eprintln!("ERROR: Model has no token embeddings.");
            std::process::exit(1);
        }
    };

    // Run spectral diagnostics
    eprintln!("Running attention spectral analysis...");
    let t_diag = Instant::now();
    let report = model.diagnose_attention(&embeddings, seq_len)?;
    let diag_ms = t_diag.elapsed().as_secs_f64() * 1000.0;

    match report {
        Some(report) => {
            eprintln!("\n=== Spectral Diagnostics Report ===");
            eprintln!("Model: {}", report.model);
            eprintln!("Run ID: {}", report.run_id);
            eprintln!("Layers profiled: {}", report.num_layers);
            let total_possible = model.config.num_layers * model.config.num_heads;
            eprintln!("Heads profiled: {} / {}", report.summary.total_heads, total_possible);
            let degeneracy_pct = if report.summary.total_heads > 0 {
                report.summary.degenerate_heads as f64 / report.summary.total_heads as f64 * 100.0
            } else {
                0.0
            };
            eprintln!("Degenerate heads: {} ({:.1}%)", report.summary.degenerate_heads, degeneracy_pct);
            eprintln!("Avg effective rank: {:.2}", report.summary.avg_effective_rank);
            eprintln!("Min effective rank: {:.2}", report.summary.min_effective_rank);
            eprintln!("Max degeneracy score: {:.3}", report.summary.max_degeneracy);
            eprintln!("Analysis time: {:.2}ms", diag_ms);
            eprintln!();

            // Per-layer summary (group flat heads vec by layer)
            let mut current_layer: Option<usize> = None;
            for head in &report.heads {
                if Some(head.layer) != current_layer {
                    current_layer = Some(head.layer);
                    // Count degenerate heads in this layer
                    let layer_heads: Vec<&_> = report.heads.iter()
                        .filter(|h| h.layer == head.layer && h.degeneracy_score >= 0.3)
                        .collect();
                    if !layer_heads.is_empty() {
                        eprintln!("Layer {:>2}: {} suspect/degenerate head(s)", head.layer, layer_heads.len());
                    }
                }
                if head.degeneracy_score >= 0.3 {
                    let sev = if head.degeneracy_score >= 0.7 { "SEV" } else { "MOD" };
                    eprintln!("  Head {:>2}: λ_max={:.3} eff_rank={:.1} MP_dist={:.4} deg={:.3} [{}]",
                        head.head,
                        head.spectral_norm,
                        head.effective_rank,
                        head.mp_distance,
                        head.degeneracy_score,
                        sev);
                }
            }
        }
        None => {
            eprintln!("No spectral data collected. Diagnostics may be disabled.");
        }
    }

    Ok(())
}