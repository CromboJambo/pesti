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
