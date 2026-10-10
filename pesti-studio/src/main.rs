//! PESTI Studio — hardware-aware model management and auto-tuning CLI
//!
//! Provides Unsloth-style UX: detect hardware, recommend configs, manage models.

use anyhow::{Result, anyhow};
use clap::{Parser, Subcommand};
use pesti_runner::llama::{LlamaRunner, ModelInfo};
use std::time::Instant;

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

fn estimate_gpu_layers(vram_gb: f64, model_size_gb: f64) -> i32 {
    let available_for_model = vram_gb * 0.85;
    if available_for_model >= model_size_gb {
        return -1;
    }
    let ratio = available_for_model / model_size_gb;
    (ratio * 32.0) as i32
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

            // Estimate model size (would be more sophisticated in real impl)
            let model_size_gb = 1.0; // Placeholder

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

            // Build runner with recommended config
            let builder = LlamaRunner::builder(&run_args.model)
                .n_ctx(config.n_ctx as u32)
                .n_threads(config.n_threads as i32)
                .n_gpu_layers(config.n_gpu_layers);

            // ... rest of inference logic
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
    }

    Ok(())
}
