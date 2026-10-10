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
    let runner = LlamaRunner::builder(&args.model)
        .n_ctx(args.ctx_size)
        .n_threads(args.threads)
        .seed(seed)
        .n_gpu_layers(args.gpu_layers)
        .build()?;
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