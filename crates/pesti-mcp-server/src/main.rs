use pesti_runner::diagnostics::spectral::SpectralProfiler;
use pesti_structural_tokenizer::StructuralTokenizer;
use rmcp::{handler::server::wrapper::Parameters, schemars, tool, tool_router};

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct FileRequest {
    pub path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SpectralAnalysisRequest {
    /// 2D attention matrix as [query_positions][key_positions]
    pub matrix: Vec<Vec<f64>>,
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

    #[tool(description = "Analyze attention matrix for spectral degeneracy using RMT bounds. Takes a square 2D attention matrix (query x key) and returns effective rank, participation ratio, Marchenko-Pastur bounds, and severity assessment.")]
    fn spectral_analyze(&self, params: Parameters<SpectralAnalysisRequest>) -> String {
        let request = &params.0;
        let n = request.matrix.len();
        if n == 0 {
            return "Error: empty attention matrix".to_string();
        }
        let m = request.matrix[0].len();
        if m != n {
            return format!("Error: attention matrix must be square (got {}x{})", n, m);
        }

        // Convert to ndarray Array2<f32>
        use ndarray::Array2;
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                data[i * n + j] = request.matrix[i][j] as f32;
            }
        }
        let attention = Array2::from_shape_vec((n, n), data).unwrap();

        // Analyze using RMT-based spectral profiler
        let profiler = SpectralProfiler::new();
        let spectrum = profiler.analyze_attention(0, 0, &attention);
        serde_json::to_string_pretty(&spectrum).unwrap_or_else(|e| format!("Serialization error: {}", e))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let server = PestiMcpServer;

    eprintln!("PESTI MCP Server starting...");
    let running = rmcp::serve_server(server, (tokio::io::stdin(), tokio::io::stdout())).await?;
    running.waiting().await?;

    Ok(())
}
