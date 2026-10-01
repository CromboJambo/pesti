// project-analyzer: Use pesti structural tokenizer to classify and organize projects
// Unix philosophy: one tool per job, compose via pipes, flat structure

use std::process::{Command, Stdio};
use std::fs;
use std::path::Path;

fn tokenize_file(path: &str) -> Option<String> {
    let output = Command::new("./target/release/struct-tok")
        .args(["tokenize", path])
        .output()
        .ok()?;
    
    if !output.status.success() { return None; }
    Some(String::from_utf8_lossy(&output.stdout).to_string())
}

fn classify_project(dir: &str) -> String {
    // Look for Cargo.toml, package.json, etc. to determine project type
    let cargo_toml = format!("{}/Cargo.toml", dir);
    if Path::new(&cargo_toml).exists() {
        return "rust-workspace".to_string();
    }
    
    let package_json = format!("{}/package.json", dir);
    if Path::new(&package_json).exists() {
        return "node-package".to_string();
    }
    
    "unknown".to_string()
}

fn analyze_directory(dir: &str) {
    println!("Analyzing: {}", dir);
    
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("  Error reading dir: {}", e);
            return;
        }
    };
    
    let mut has_source = false;
    let mut has_config = false;
    let mut has_docs = false;
    let mut files_count = 0;
    
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path();
        
        if path.is_dir() {
            // Skip hidden dirs and build artifacts
            if name.starts_with('.') || name == "target" || name == "node_modules" {
                continue;
            }
        } else {
            files_count += 1;
            
            // Check file type via extension
            let ext = path.extension().map(|e| e.to_string_lossy().to_string());
            
            match ext.as_deref() {
                Some("rs") | Some("toml") => has_source = true,
                Some("json") | Some("yaml") | Some("yml") => has_config = true,
                Some("md") | Some("txt") => has_docs = true,
                _ => {}
            }
        }
    }
    
    println!("  Type: {}", classify_project(dir));
    println!("  Files: {}", files_count);
    println!("  Has source: {}, config: {}, docs: {}", has_source, has_config, has_docs);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        println!("Usage: project-analyzer <directory>");
        return;
    }
    
    let dir = &args[1];
    analyze_directory(dir);
}