use pesti_structural_tokenizer::StructuralTokenizer;
use rmcp::{handler::server::wrapper::Parameters, schemars, tool, tool_router};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct FileRequest {
    pub path: String,
}

#[derive(Debug, Clone)]
pub struct PestiMcpServer;

#[tool_router(server_handler)]
impl PestiMcpServer {
    #[tool(description = "Tokenize Rust source file into structural tokens")]
    fn tokenize_file(&self, params: Parameters<FileRequest>) -> String {
        let path = &params.0.path;
        let tokenizer = StructuralTokenizer::new();
        match std::fs::read_to_string(path) {
            Ok(content) => {
                match tokenizer.tokenize(&content) {
                    Ok(tokens) => {
                        let mut results = Vec::new();
                        for token in &tokens {
                            results.push(format!("{:?}: {}", token.kind, token.text));
                        }
                        results.join("\n")
                    }
                    Err(e) => format!("Tokenization error: {}", e),
                }
            }
            Err(e) => format!("Error reading {}: {}", path, e),
        }
    }

    #[tool(description = "List all function declarations in a Rust file")]
    fn list_functions(&self, params: Parameters<FileRequest>) -> String {
        let path = &params.0.path;
        let tokenizer = StructuralTokenizer::new();
        match std::fs::read_to_string(path) {
            Ok(content) => {
                match tokenizer.tokenize(&content) {
                    Ok(tokens) => {
                        let mut functions = Vec::new();
                        for token in &tokens {
                            if matches!(token.kind, pesti_structural_tokenizer::TokenKind::FnDecl) {
                                functions.push(token.text.clone());
                            }
                        }
                        functions.join("\n")
                    }
                    Err(e) => format!("Tokenization error: {}", e),
                }
            }
            Err(e) => format!("Error reading {}: {}", path, e),
        }
    }

    #[tool(description = "Analyze code structure and return token distribution")]
    fn analyze_structure(&self, params: Parameters<FileRequest>) -> String {
        let path = &params.0.path;
        let tokenizer = StructuralTokenizer::new();
        match std::fs::read_to_string(path) {
            Ok(content) => {
                match tokenizer.tokenize(&content) {
                    Ok(tokens) => {
                        let mut distribution: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                        for token in &tokens {
                            *distribution.entry(format!("{:?}", token.kind)).or_insert(0) += 1;
                        }
                        serde_json::to_string(&distribution).unwrap_or_default()
                    }
                    Err(e) => format!("Tokenization error: {}", e),
                }
            }
            Err(e) => format!("Error reading {}: {}", path, e),
        }
    }

    #[tool(description = "Extract API surface (public functions and structs) from Rust file")]
    fn extract_api_surface(&self, params: Parameters<FileRequest>) -> String {
        let path = &params.0.path;
        let tokenizer = StructuralTokenizer::new();
        match std::fs::read_to_string(path) {
            Ok(content) => {
                match tokenizer.tokenize(&content) {
                    Ok(tokens) => {
                        let mut api_items = Vec::new();
                        for token in &tokens {
                            if matches!(token.kind, pesti_structural_tokenizer::TokenKind::FnDecl | pesti_structural_tokenizer::TokenKind::StructDecl | pesti_structural_tokenizer::TokenKind::EnumDecl) {
                                api_items.push(format!("{} ({:?})", token.text, token.kind));
                            }
                        }
                        api_items.join("\n")
                    }
                    Err(e) => format!("Tokenization error: {}", e),
                }
            }
            Err(e) => format!("Error reading {}: {}", path, e),
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let server = PestiMcpServer;

    eprintln!("PESTI MCP Server starting...");
    rmcp::serve_server(server, (tokio::io::stdin(), tokio::io::stdout())).await?;

    Ok(())
}
