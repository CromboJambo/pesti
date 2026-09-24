use std::env;
use std::fs;
use std::path::PathBuf;
use serde_json::json;

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 2 {
        eprintln!("Usage: hermes-struct-tok-pipeline watch <dir> | process <file>");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "watch" => {
            let watch_dir = PathBuf::from(&args[2]);
            println!("Watching {} for Rust source files...", watch_dir.display());
            process_directory(&watch_dir);
        }
        "process" => {
            let file = &args[2];
            process_file(file);
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }
}

fn process_directory(dir: &PathBuf) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "rs" {
                    if path.file_name().is_some() {
                        process_file(path.to_str().unwrap());
                    }
                }
            }
        }
    }
}

fn process_file(file: &str) {
    println!("Processing: {}", file);
    let output = std::process::Command::new("struct-tok")
        .args(["encode", "--file"])
        .arg(file)
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let tokens = String::from_utf8_lossy(&o.stdout);
            let event = json!({
                "type": "rust_source",
                "file": file,
                "tokens": tokens.trim()
            });
            println!("{}", event.to_string());
        }
        _ => {
            eprintln!("Failed to tokenize {}", file);
        }
    }
}
