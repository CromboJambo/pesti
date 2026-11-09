//! pesti-embed: Embedding model CLI tool.
//!
//! Convert text to vector embeddings using PESTI's transformer inference engine.

use anyhow::Result;
use clap::{Parser, Subcommand};
use pesti_embeddings::{Embedder, PoolingStrategy};

#[derive(Parser)]
#[command(name = "pesti-embed", about = "PESTI embedding model CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Embed a single text string
    Embed {
        /// Path to GGUF model file
        model: String,
        /// Text to embed
        text: String,
        /// Pooling strategy (mean, first, last)
        #[arg(long, default_value = "mean")]
        pooling: String,
        /// Don't L2-normalize output
        #[arg(long)]
        no_normalize: bool,
    },
    /// Compute similarity between two texts
    Similar {
        /// Path to GGUF model file
        model: String,
        /// First text
        text1: String,
        /// Second text
        text2: String,
    },
    /// Embed multiple texts from a file (one per line)
    Batch {
        /// Path to GGUF model file
        model: String,
        /// Input file (one text per line)
        input: String,
        /// Output file for embeddings (JSON array)
        #[arg(long)]
        output: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Embed {
            model,
            text,
            pooling,
            no_normalize,
        } => {
            let config = pesti_embeddings::EmbeddingConfig {
                model_path: model,
                tokenizer_path: None,
                pooling: match pooling.as_str() {
                    "mean" => PoolingStrategy::Mean,
                    "first" => PoolingStrategy::First,
                    "last" => PoolingStrategy::Last,
                    _ => return Err(anyhow::anyhow!("Unknown pooling strategy: {}", pooling)),
                },
                normalize: !no_normalize,
                max_seq_len: 512,
            };

            let embedder = Embedder::load_with_config(&config)?;
            let embedding = embedder.embed(&text)?;

            println!("Embedding dimension: {}", embedding.len());
            println!("{}", serde_json::to_string(&embedding)?);
        }

        Commands::Similar {
            model,
            text1,
            text2,
        } => {
            let embedder = Embedder::load(&model)?;
            let emb1 = embedder.embed(&text1)?;
            let emb2 = embedder.embed(&text2)?;

            let sim = pesti_embeddings::cosine_similarity(&emb1, &emb2);
            println!("Cosine similarity: {:.4}", sim);
        }

        Commands::Batch {
            model,
            input,
            output,
        } => {
            let embedder = Embedder::load(&model)?;

            let content = std::fs::read_to_string(input)?;
            let texts: Vec<&str> = content.lines().collect();

            let embeddings = embedder.embed_batch(&texts)?;

            if let Some(output_path) = output {
                let json = serde_json::to_string_pretty(&embeddings)?;
                std::fs::write(&output_path, json)?;
                println!("Wrote {} embeddings to {}", embeddings.len(), output_path);
            } else {
                println!("{}", serde_json::to_string_pretty(&embeddings)?);
            }
        }
    }

    Ok(())
}