//! PESTI Studio — hardware-aware model management and auto-tuning CLI
//!
//! Provides Unsloth-style UX: detect hardware, recommend configs, manage models.

use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand};
use pesti_runner::llama::{LlamaRunner, ModelInfo};
use std::time::Instant;

mod benchmark;

#[derive(Parser)]
#[command(
    name = "pesti-studio",
    about = "PESTI Studio — hardware-aware LLM inference"
)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect hardware and recommend configuration
    Detect(DetectArgs),
    /// Run inference with auto-tuning
    Run(RunArgs),
    /// List available local models
    Models(ListModelsArgs),
    /// Download a model from HuggingFace Hub
    Download(DownloadArgs),
    /// Search for models on HuggingFace Hub
    Search(SearchArgs),
    /// Benchmark inference performance across configurations
    Benchmark(BenchmarkArgs),
}

#[derive(Parser)]
struct BenchmarkArgs {
    /// Model path to benchmark
    model_path: String,
    /// GPU layers to use (-1 = auto)
    #[arg(long = "gpu-layers", default_value_t = -1)]
    n_gpu_layers: i32,
}

#[derive(Parser)]
struct SearchArgs {
    /// Search query (e.g., "llama 7B instruct")
    query: String,
    /// Maximum results to show
    #[arg(long = "limit", default_value = "10")]
    limit: usize,
}

#[derive(Parser)]
struct DownloadArgs {
    /// HuggingFace repo ID (e.g., Qwen/Qwen2.5-0.5B-Instruct-GGUF)
    repo_id: String,
    /// Filename to download (default: auto-detect)
    #[arg(long = "file")]
    filename: Option<String>,
}

#[derive(Parser)]
struct DetectArgs {
    /// Show verbose output including raw metrics
    #[arg(long)]
    verbose: bool,
}

#[derive(Parser)]
struct RunArgs {
    /// Model path or HuggingFace repo ID
    model: String,
    /// Prompt text
    #[arg(short = 'p', long = "prompt")]
    prompt: Option<String>,
    /// Auto-tune based on hardware (default: true)
    #[arg(long, default_value_t = true)]
    auto_tune: bool,
    /// Number of tokens to generate
    #[arg(short = 'n', long = "n-predict", default_value = "256")]
    n_predict: i32,
}

#[derive(Parser)]
struct ListModelsArgs {
    /// Include remote HuggingFace models (requires network)
    #[arg(long)]
    remote: bool,
}

fn detect_hardware() -> Result<HardwareInfo> {
    let sys = sysinfo::System::new_with_specifics(sysinfo::RefreshKind::everything());

    let cpu_count = sys.cpus().len();
    let total_memory_gb = sys.total_memory() as f64 / (1024.0 * 1024.0 * 1024.0);

    // Try to detect GPU via CUDA runtime
    let gpu_info = detect_cuda_gpu()?;

    Ok(HardwareInfo {
        cpu_count,
        total_memory_gb,
        gpu: gpu_info,
    })
}

fn detect_cuda_gpu() -> Result<Option<GpuInfo>> {
    // Use pesti-runner's CUDA detection via llama.cpp FFI
    // This will probe for available GPUs through the library
    let result = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader"])
        .output();

    match result {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let first_line = stdout.lines().next();

            if let Some(line) = first_line {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 2 {
                    let name = parts[0].trim().to_string();
                    // Parse memory (e.g., "8192 MiB")
                    let mem_str = parts[1].trim();
                    let mem_parts: Vec<&str> = mem_str.split(' ').collect();
                    if mem_parts.len() >= 2 {
                        if let Ok(mem_mb) = mem_parts[0].parse::<f64>() {
                            let vram_gb = mem_mb / 1024.0;
                            return Ok(Some(GpuInfo { name, vram_gb }));
                        }
                    }
                }
            }
        }
        _ => {}
    }

    // Fallback: try to load a dummy model and see what GPU layers it reports
    Ok(None)
}

#[derive(Debug)]
struct HardwareInfo {
    cpu_count: usize,
    total_memory_gb: f64,
    gpu: Option<GpuInfo>,
}

#[derive(Debug)]
struct GpuInfo {
    name: String,
    vram_gb: f64,
}

fn recommend_config(hw: &HardwareInfo, model_size_gb: f64) -> InferenceConfig {
    let config = if let Some(gpu) = &hw.gpu {
        // GPU available — calculate optimal offloading
        let available_vram = gpu.vram_gb * 0.9; // 10% overhead

        if available_vram > model_size_gb * 2.5 {
            // Plenty of VRAM — full GPU, large context
            InferenceConfig {
                n_gpu_layers: -1,
                n_ctx: 8192,
                n_batch: 32,
                n_threads: hw.cpu_count,
                quantization: "Q5_K_M".to_string(),
                strategy: "GPU full model + large context".to_string(),
            }
        } else if available_vram > model_size_gb * 1.5 {
            // Fit model, limited context
            InferenceConfig {
                n_gpu_layers: -1,
                n_ctx: 4096,
                n_batch: 8,
                n_threads: hw.cpu_count / 2,
                quantization: "Q4_K_M".to_string(),
                strategy: "GPU full model, moderate context".to_string(),
            }
        } else {
            // Need CPU offloading
            let gpu_layers = estimate_gpu_layers(gpu.vram_gb, model_size_gb);
            InferenceConfig {
                n_gpu_layers: gpu_layers,
                n_ctx: 2048,
                n_batch: 1,
                n_threads: hw.cpu_count,
                quantization: "Q3_K_M".to_string(),
                strategy: format!("Hybrid GPU({} layers)/CPU", gpu_layers),
            }
        }
    } else {
        // CPU only
        InferenceConfig {
            n_gpu_layers: 0,
            n_ctx: 2048,
            n_batch: 1,
            n_threads: hw.cpu_count,
            quantization: "Q4_K_M".to_string(),
            strategy: "CPU only".to_string(),
        }
    };

    config
}

fn list_local_models() -> Vec<std::path::PathBuf> {
    let mut models = Vec::new();
    let search_dirs = [
        "/home/crombo/.local/share/pesti/models",
        "/home/crombo/projects/pesti/conformance-corpus",
        "/models",
    ];

    for dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |e| e == "gguf") {
                    models.push(path);
                }
            }
        }
    }

    models
}

fn download_model(repo_id: &str, filename: Option<&str>) -> Result<std::path::PathBuf> {
    println!("Downloading from HuggingFace Hub: {}", repo_id);

    let path = pesti_runner::runtime::Runtime::download_from_hf(repo_id, filename.unwrap_or(""))?;
    println!("Downloaded to: {}", path.display());
    Ok(path)
}

fn search_hf_models(query: &str, limit: usize) -> Result<Vec<serde_json::Value>> {
    let url = format!(
        "https://huggingface.co/api/models?search={}&sort=downloads&direction=-1&limit={}",
        query.replace(' ', "+"),
        limit
    );

    let client = reqwest::blocking::Client::new();
    let response = client.get(&url).send()?;

    if !response.status().is_success() {
        return Err(anyhow!("HF API request failed: {}", response.status()));
    }

    let models: Vec<serde_json::Value> = response.json()?;
    Ok(models)
}

fn estimate_gpu_layers(vram_gb: f64, model_size_gb: f64) -> i32 {
    let available_for_model = vram_gb * 0.85;
    if available_for_model >= model_size_gb {
        return -1;
    }
    let ratio = available_for_model / model_size_gb;
    (ratio * 32.0) as i32
}

fn estimate_model_size(model_path: &str) -> f64 {
    // Rough heuristic based on model name patterns
    if model_path.contains("0.5B") || model_path.contains("0.5b") {
        return 1.0;
    } else if model_path.contains("1.5B") || model_path.contains("1.5b") {
        return 3.0;
    } else if model_path.contains("3B") || model_path.contains("3b") {
        return 6.0;
    } else if model_path.contains("7B") || model_path.contains("7b") {
        return 14.0;
    } else if model_path.contains("13B") || model_path.contains("13b") {
        return 28.0;
    } else if model_path.contains("30B") || model_path.contains("30b") {
        return 60.0;
    } else if model_path.contains("70B") || model_path.contains("70b") {
        return 140.0;
    }
    // Default assumption
    14.0
}

#[derive(Debug)]
struct InferenceConfig {
    n_gpu_layers: i32,
    n_ctx: u32,
    n_batch: u32,
    n_threads: usize,
    quantization: String,
    strategy: String,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    match args.command {
        Commands::Detect(detect_args) => {
            println!("PESTI Studio — Hardware Detection");
            println!("=================================");

            let hw = detect_hardware()?;

            println!("CPU: {} cores", hw.cpu_count);
            println!("RAM: {:.1} GB", hw.total_memory_gb);

            if let Some(gpu) = &hw.gpu {
                println!("GPU: {} ({:.1} GB VRAM)", gpu.name, gpu.vram_gb);
            } else {
                println!("GPU: Not detected");
            }

            if detect_args.verbose {
                println!("\nDetailed metrics available via --verbose flag");
            }
        }

        Commands::Run(run_args) => {
            println!("PESTI Studio — Running Inference");
            println!("=================================");

            let hw = detect_hardware()?;

            // Estimate model size from path (rough heuristic)
            let model_size_gb = estimate_model_size(&run_args.model);

            let config = recommend_config(&hw, model_size_gb);

            println!(
                "Hardware: {} cores, {:.1} GB RAM",
                hw.cpu_count, hw.total_memory_gb
            );
            if let Some(gpu) = &hw.gpu {
                println!("GPU: {} ({:.1} GB VRAM)", gpu.name, gpu.vram_gb);
            }

            println!("\nRecommended config:");
            println!("  Strategy: {}", config.strategy);
            println!("  GPU layers: {}", config.n_gpu_layers);
            println!("  Context window: {}", config.n_ctx);
            println!("  Batch size: {}", config.n_batch);
            println!("  Threads: {}", config.n_threads);
            println!("  Quantization: {}", config.quantization);

            // Get prompt (from args or stdin)
            let prompt = if let Some(p) = run_args.prompt {
                p
            } else {
                use std::io::Write;
                print!("Prompt: ");
                let mut out = std::io::stdout();
                out.flush()?;
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                input.trim().to_string()
            };

            println!("\n--- Generating ---");

            // Build runner with recommended config
            let builder = LlamaRunner::builder(&run_args.model)
                .n_ctx(config.n_ctx as u32)
                .n_threads(config.n_threads as i32)
                .n_gpu_layers(config.n_gpu_layers);

            let runner = builder.build()?;

            // Run inference with timing
            let start = Instant::now();
            let sampling = pesti_runner::llama::SamplingConfig {
                temperature: 0.7,
                top_k: 40,
                top_p: 0.95,
                max_tokens: run_args.n_predict as u32,
                ..Default::default()
            };
            let result = runner.generate(&prompt, &sampling)?;

            let elapsed = start.elapsed();
            println!("{}", result.text);
            println!("\n--- Done ---");
            println!(
                "Generated {} tokens in {:.2}s ({:.1} tok/s)",
                result.generated_tokens,
                elapsed.as_secs_f64(),
                result.generated_tokens as f64 / elapsed.as_secs_f64()
            );
        }

        Commands::Models(list_args) => {
            println!("PESTI Studio — Model Management");
            println!("=================================");

            if list_args.remote {
                println!("Remote model listing (HuggingFace Hub) coming soon...");
            } else {
                let models = list_local_models();
                if models.is_empty() {
                    println!("No local GGUF models found.");
                    println!("Run: pesti-studio download <repo-id> --file <filename>");
                } else {
                    println!("Local models ({}):", models.len());
                    for model in &models {
                        println!("  {}", model.display());
                    }
                }
            }
        }

        Commands::Download(download_args) => {
            println!("PESTI Studio — Downloading Model");
            println!("=================================");

            let path = download_model(&download_args.repo_id, download_args.filename.as_deref())?;
            println!("Model downloaded to: {}", path.display());
        }

        Commands::Search(search_args) => {
            println!("PESTI Studio — Searching HuggingFace Hub");
            println!("==========================================");
            println!("Query: {}", search_args.query);

            let models = search_hf_models(&search_args.query, search_args.limit)?;

            if models.is_empty() {
                println!("No results found.");
            } else {
                for (i, model) in models.iter().enumerate() {
                    let id = model["modelId"].as_str().unwrap_or("unknown");
                    let downloads = model["downloads"].as_u64().unwrap_or(0);
                    println!("{}. {} ({} downloads)", i + 1, id, downloads);
                }
            }
        }

        Commands::Benchmark(bench_args) => {
            println!("PESTI Studio — Benchmarking");
            println!("=================================");
            println!("Model: {}", bench_args.model_path);
            println!();

            let results =
                benchmark::run_benchmarks(&bench_args.model_path, bench_args.n_gpu_layers)?;
            benchmark::print_results(&results);
        }
    }

    Ok(())
}
