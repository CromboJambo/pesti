use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

fn analyze_file(path: &Path) -> Result<(usize, HashMap<String, usize>, Vec<String>), String> {
    let content = fs::read_to_string(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    let tokenizer = StructuralTokenizer::new();
    let tokens = tokenizer.tokenize(&content)
        .map_err(|e| format!("{}: {}", path.display(), e))?;

    let mut kind_counts: HashMap<String, usize> = HashMap::new();
    for tok in &tokens {
        *kind_counts.entry(format!("{:?}", tok.kind)).or_insert(0) += 1;
    }

    // Collect function names from FnDecl tokens
    let fn_names: Vec<String> = tokens.iter()
        .filter(|t| matches!(t.kind, TokenKind::FnDecl))
        .map(|t| {
            let text = &t.text;
            // Extract function name from "fn name(..." or "pub fn name(..."
            if let Some(pos) = text.find("fn ") {
                let rest = &text[pos + 3..];
                if let Some(end) = rest.find('(') {
                    return rest[..end].trim().to_string();
                }
            }
            "unknown".to_string()
        })
        .collect();

    Ok((tokens.len(), kind_counts, fn_names))
}

fn main() {
    let root = Path::new("/tmp/herdr");
    
    // Find all .rs files (excluding target/)
    let mut files: Vec<PathBuf> = vec![];
    for entry in walkdir(root) {
        if let Ok(path) = entry {
            if path.path().extension() == Some("rs") 
                && !path.path().to_string_lossy().contains("/target/") {
                files.push(path.into_path());
            }
        }
    }

    println!("Analyzing {} Rust files in herdr...", files.len());
    
    let mut total_tokens = 0;
    let mut total_fn_decls = 0;
    let mut total_struct_decls = 0;
    let mut total_enum_decls = 0;
    let mut total_impl_blocks = 0;
    let mut all_fn_names: Vec<String> = vec![];

    for path in &files {
        match analyze_file(path) {
            Ok((tokens, kinds, fn_names)) => {
                total_tokens += tokens;
                total_fn_decls += kinds.get("FnDecl").copied().unwrap_or(0);
                total_struct_decls += kinds.get("StructDecl").copied().unwrap_or(0);
                total_enum_decls += kinds.get("EnumDecl").copied().unwrap_or(0);
                total_impl_blocks += kinds.get("ImplBlock").copied().unwrap_or(0);
                all_fn_names.extend(fn_names);
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }

    println!("\n=== herdr codebase structural analysis ===");
    println!("Files analyzed: {}", files.len());
    println!("Total structural tokens: {}", total_tokens);
    println!("Functions declared: {}", total_fn_decls);
    println!("Structs declared: {}", total_struct_decls);
    println!("Enums declared: {}", total_enum_decls);
    println!("Impl blocks: {}", total_impl_blocks);
    
    // Sample some function names for context
    println!("\nSample functions (first 30):");
    for name in all_fn_names.iter().take(30) {
        println!("  - {}", name);
    }
}
