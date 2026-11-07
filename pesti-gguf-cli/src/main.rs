use clap::Parser;
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(
    name = "gguf-inspect",
    about = "Inspect GGUF model file headers and metadata"
)]
struct Args {
    /// Path to the GGUF file
    #[arg(name = "path")]
    gguf_path: PathBuf,

    /// List tensor names only (no details)
    #[arg(short, long, default_value_t = false)]
    tensors_only: bool,

    /// Extract a specific tensor as raw bytes (hex dump first 64 bytes)
    #[arg(short, long)]
    extract: Option<String>,

    /// Dump the full parsed header as JSON (for conformance comparison)
    #[arg(long, default_value_t = false)]
    json: bool,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .without_time()
        .init();

    let args = Args::parse();

    if !args.gguf_path.exists() {
        eprintln!("error: file not found: {}", args.gguf_path.display());
        std::process::exit(1);
    }

    let header = match pesti_gguf::parser::parse_gguf(&args.gguf_path) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("error: failed to parse GGUF: {e}");
            std::process::exit(1);
        }
    };

    if args.json {
        dump_json(&header);
        return;
    }

    print_header_summary(&header);

    if args.tensors_only {
        print_tensor_names(&header);
        return;
    }

    print_tensors(&header);
    handle_extract(&args, &header);
}

fn dump_json(header: &pesti_gguf::GgufHeader) {
    let mut out = serde_json::Map::new();
    match serde_json::to_value(header) {
        Ok(serde_json::Value::Object(map)) => out = map,
        Ok(_) => {
            eprintln!("error: header serialized to non-object value");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("error: failed to serialize header: {e}");
            std::process::exit(1);
        }
    }

    let stored: Vec<serde_json::Value> = header
        .tensors
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "stored_size": t.stored_size().map_err(|e| e.to_string()).ok(),
            })
        })
        .collect();
    out.insert("tensors_stored_size".to_string(), serde_json::Value::Array(stored));

    match serde_json::to_string_pretty(&out) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("error: failed to serialize header: {e}");
            std::process::exit(1);
        }
    }
}

fn print_header_summary(header: &pesti_gguf::GgufHeader) {
    println!("GGUF v{}", header.version);
    if let Some(alignment) = header.data_alignment {
        println!("  data alignment: {}", alignment);
    }
    println!("  tensors: {}", header.tensors.len());
    println!("  kv pairs: {}", header.kv_pairs.len());
    println!();

    if let Some(arch) = header.architecture() {
        println!("  architecture: {}", arch);
    }
    if let Some(ft) = header.get_kv_str("general.file_type") {
        println!("  file type: {}", ft);
    }
    println!();

    print_config(header);
}

fn print_config(header: &pesti_gguf::GgufHeader) {
    let common_keys = [
        "llama.context_length",
        "llama.embedding_length",
        "llama.block_count",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.rope.dimension_count",
        "llama.attention.layer_norm_rms_epsilon",
        "llama.feed_forward_length",
        "llama.vocab_size",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
        "general.architecture",
        "general.file_type",
        "rope.scaling.type",
        "rope.scaling.factor",
    ];

    println!("  config:");
    for key in &common_keys {
        if let Some(val) = header.get_kv_str(key) {
            println!("    {}: {}", key, val);
        } else if let Some(val) = header.get_kv_u32(key) {
            println!("    {}: {}", key, val);
        }
    }
    println!();
}

fn print_tensor_names(header: &pesti_gguf::GgufHeader) {
    for tensor in &header.tensors {
        println!("  {}", tensor.name);
    }
}

fn print_tensors(header: &pesti_gguf::GgufHeader) {
    if header.tensors.is_empty() {
        return;
    }

    println!("  tensors:");
    for tensor in &header.tensors {
        let shape_str = tensor
            .shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join("x");
        let dtype = pesti_gguf::GgufDtype::from_u32(tensor.dtype);
        println!(
            "    {:60}  [{}]  dtype={:<6}  stored={:>12}  offset={}",
            tensor.name,
            shape_str,
            dtype.name(),
            tensor.stored_size().unwrap_or(0),
            tensor.offset
        );
    }
    println!();
}

fn handle_extract(args: &Args, header: &pesti_gguf::GgufHeader) {
    let Some(tensor_name) = &args.extract else { return };

    let Some(tensor) = header.tensors.iter().find(|t| t.name == *tensor_name) else {
        eprintln!("  error: tensor '{}' not found", tensor_name);
        println!("  available tensors:");
        for tensor in &header.tensors {
            println!("    - {}", tensor.name);
        }
        return;
    };

    let stored = tensor.stored_size().unwrap_or(0);
    println!(
        "  extracting {} ({} bytes stored, {} dequantized)...",
        tensor_name,
        stored,
        tensor.element_count()
    );

    match pesti_gguf::parser::extract_tensor_bytes_from_path(
        &args.gguf_path,
        tensor.offset,
        stored as usize,
    ) {
        Ok(bytes) => print_extracted(&bytes),
        Err(e) => eprintln!("  error: failed to extract tensor: {e}"),
    }
}

fn print_extracted(bytes: &[u8]) {
    println!("  extracted {} bytes", bytes.len());
    let hex_len = bytes.len().min(64);
    let hex = bytes[..hex_len]
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect::<Vec<_>>()
        .join(" ");
    println!("  first {} bytes (hex): {}", hex_len, hex);
    if bytes.len() > 64 {
        println!("  ... ({} more bytes)", bytes.len() - 64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn args_debug_impl() {
        let args = Args {
            gguf_path: PathBuf::from("/tmp/test.gguf"),
            tensors_only: false,
            extract: None,
            json: false,
        };
        let debug_str = format!("{:?}", args);
        assert!(debug_str.contains("gguf_path"));
        assert!(debug_str.contains("tensors_only"));
        assert!(debug_str.contains("extract"));
    }

    #[test]
    fn args_clone_behavior() {
        let args = Args {
            gguf_path: PathBuf::from("/tmp/test.gguf"),
            tensors_only: true,
            extract: Some("tensor1".to_string()),
            json: false,
        };
        let cloned = Args {
            gguf_path: args.gguf_path.clone(),
            tensors_only: args.tensors_only,
            extract: args.extract.clone(),
            json: false,
        };
        assert!(cloned.tensors_only);
        assert_eq!(cloned.extract, Some("tensor1".to_string()));
    }

    #[test]
    fn args_path_exists_false() {
        let args = Args {
            gguf_path: PathBuf::from("/nonexistent/path/that/does/not/exist.gguf"),
            tensors_only: false,
            extract: None,
            json: false,
        };
        assert!(!args.gguf_path.exists());
    }

    #[test]
    fn args_path_exists_true() {
        let args = Args {
            gguf_path: PathBuf::from("/tmp"),
            tensors_only: false,
            extract: None,
            json: false,
        };
        assert!(args.gguf_path.exists());
    }
}
