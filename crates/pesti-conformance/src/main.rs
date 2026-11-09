//! PESTI Conformance Test Runner Binary

use clap::{Parser, Subcommand};
use pesti_conformance::ConformanceConfig;

#[derive(Parser)]
#[command(name = "pesti-conformance")]
#[command(about = "Differential conformance testing against reference implementations", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run conformance tests on a model corpus
    Run {
        /// Path to GGUF model corpus directory
        #[arg(short, long, default_value = "./models/")]
        corpus_dir: std::path::PathBuf,
        /// Reference llama.cpp binary path (optional)
        #[arg(long)]
        reference_llama_cpp: Option<std::path::PathBuf>,
        /// Minimum passing count threshold
        #[arg(long, default_value = "0")]
        floor_pass_count: usize,
        /// Floor file for CI gating (optional)
        #[arg(long)]
        floor_file: Option<std::path::PathBuf>,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Run {
            corpus_dir,
            reference_llama_cpp,
            floor_pass_count,
            floor_file,
        } => {
            let config = ConformanceConfig {
                corpus_dir: corpus_dir.clone(),
                reference_llama_cpp: reference_llama_cpp.clone(),
                floor_pass_count: *floor_pass_count,
                floor_file: floor_file.clone(),
            };

            match pesti_conformance::run_conformance(&config) {
                Ok(result) => {
                    println!(
                        "Conformance complete: {}/{} passed ({:.1}%)",
                        result.passed.len(),
                        result.total_models,
                        (result.passed.len() as f64 / result.total_models.max(1) as f64) * 100.0,
                    );

                    for failure in &result.failures {
                        eprintln!(
                            "FAILURE: {} - expected={} actual={}",
                            failure.model_name, failure.expected_hash, failure.actual_hash
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Error: {:?}", e);
                    std::process::exit(1);
                }
            }
        }
    }

    Ok(())
}
