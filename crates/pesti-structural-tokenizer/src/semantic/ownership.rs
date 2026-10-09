//! Ownership mode analysis for Rust source code.
//! Lightweight string-based scanner to detect ownership patterns.

use std::collections::HashMap;

/// Ownership mode inferred for a variable binding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipMode {
    Owned,           // value moved in (default)
    Borrowed,        // &T
    MutablyBorrowed, // &mut T
}

/// Annotation attached to AST nodes by the ownership analyzer
#[derive(Debug, Clone)]
pub struct OwnershipAnnotation {
    pub var_name: String,
    pub mode: OwnershipMode,
    /// Confidence: 0=unknown, 1=inferred, 2=proven from syntax
    pub confidence: u8,
}

/// Minimal ownership analyzer — tracks binding modes through the AST
pub struct OwnershipAnalyzer {
    /// Current scope's variable -> ownership mode
    scopes: Vec<HashMap<String, OwnershipMode>>,
    /// Annotations to emit (node_id -> annotation)
    pub annotations: HashMap<usize, OwnershipAnnotation>,
}

impl OwnershipAnalyzer {
    pub fn new() -> Self {
        let mut scopes = Vec::new();
        scopes.push(HashMap::new());
        Self {
            scopes,
            annotations: HashMap::new(),
        }
    }

    /// Enter a new scope (function body, block)
    fn enter_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Exit current scope
    fn exit_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Record a variable binding with its ownership mode
    fn bind(&mut self, name: &str, mode: OwnershipMode, node_id: usize) {
        let scope = self.scopes.last_mut().unwrap();
        scope.insert(name.to_string(), mode);

        // Emit annotation for this binding site
        self.annotations.insert(
            node_id,
            OwnershipAnnotation {
                var_name: name.to_string(),
                mode,
                confidence: 2, // proven from syntax
            },
        );
    }

    /// Look up ownership mode for a variable at use site
    fn lookup(&self, name: &str) -> Option<OwnershipMode> {
        for scope in self.scopes.iter().rev() {
            if let Some(mode) = scope.get(name) {
                return Some(*mode);
            }
        }
        None
    }

    /// Analyze a function for ownership patterns (string-based scanning)
    pub fn analyze_function(&mut self, func: &str) {
        let node_id = 0; // Simplified - real impl would use actual node IDs

        // Analyze parameters by looking at function signature
        if let Some(params) = extract_params(func) {
            for (i, param) in params.iter().enumerate() {
                let trimmed = param.trim();
                let mode = if trimmed.starts_with("&mut ") {
                    OwnershipMode::MutablyBorrowed
                } else if trimmed.starts_with("&") {
                    OwnershipMode::Borrowed
                } else {
                    OwnershipMode::Owned
                };

                // Extract variable name (last part before :)
                let name = extract_param_name(trimmed);
                self.bind(&name, mode, node_id + i);
            }
        }

        // Enter function scope
        self.enter_scope();

        // Analyze body for let bindings
        for (i, line) in func.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("let ") && !trimmed.starts_with("let mut") {
                // Immutable binding - detect borrow type from type annotation
                if trimmed.contains(": &mut ") {
                    self.bind("var", OwnershipMode::MutablyBorrowed, i);
                } else if trimmed.contains(": &") {
                    self.bind("var", OwnershipMode::Borrowed, i);
                } else {
                    self.bind("var", OwnershipMode::Owned, i);
                }
            }
        }

        self.exit_scope();
    }
}

/// Extract function parameters from signature (simplified)
fn extract_params(func: &str) -> Option<Vec<String>> {
    let start = func.find('(')?;
    let end = func.find(')')?;
    if start >= end {
        return None;
    }

    let params_str = &func[start + 1..end];
    // Handle empty params
    if params_str.trim().is_empty() {
        return Some(vec![]);
    }

    // Split by comma, but be careful with nested parens (simplified)
    Some(params_str.split(',').map(|s| s.to_string()).collect())
}

/// Extract parameter name from signature
fn extract_param_name(param: &str) -> String {
    let trimmed = param.trim();
    // Remove leading reference markers
    let without_ref = trimmed
        .trim_start_matches('&')
        .trim_start_matches(' ')
        .trim_start_matches("mut ")
        .trim_start_matches(' ');

    // Get name before colon (type annotation)
    if let Some(colon_pos) = without_ref.find(':') {
        return without_ref[..colon_pos].trim().to_string();
    }

    without_ref.to_string()
}

/// Analyze ownership across an entire program (simplified: source string input).
pub fn analyze_ownership(source: &str) -> HashMap<usize, OwnershipAnnotation> {
    let mut analyzer = OwnershipAnalyzer::new();

    // Simplified: treat whole source as one function body for now.
    // Real impl would parse into functions first.
    analyzer.enter_scope();

    // Detect ownership patterns via lightweight string scanning
    for (i, line) in source.lines().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("let ") && !trimmed.starts_with("let mut") {
            // Immutable binding - likely owned or borrowed
            if trimmed.contains(": &mut ") {
                analyzer.bind("var", OwnershipMode::MutablyBorrowed, i);
            } else if trimmed.contains(": &") {
                analyzer.bind("var", OwnershipMode::Borrowed, i);
            } else {
                analyzer.bind("var", OwnershipMode::Owned, i);
            }
        }
    }

    analyzer.exit_scope();
    analyzer.annotations
}
