//! Error handling annotation layer.
//! 
//! Tracks how error conditions (Result, Option) are handled at each node:
//! propagated via ?, unwrapped, matched on, or ignored.

use std::collections::HashMap;

/// How an error condition is handled at a given point
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorHandling {
    Propagated,      // ? operator - passes error up to caller
    Unwrapped,       // .unwrap() - panics on error
    Expected,        // .expect(msg) - panics with message
    Matched,         // match { Ok(v) => ..., Err(e) => ... }
    Ignored,         // let _ = result; or dropped
    Functional,      // .ok(), .and_then(), .map() chain
    None,            // not an error-producing operation
}

/// Annotation for error handling at a specific node
#[derive(Debug, Clone)]
pub struct ErrorAnnotation {
    pub handling: ErrorHandling,
    /// If propagated, the function boundary where it exits (node_id)
    pub propagates_to: Option<usize>,
    /// Confidence in this annotation
    pub confidence: u8,
}

/// Tracks error state through expressions
pub struct ErrorAnalyzer {
    /// Current function's return type hints
    current_fn_returns_option: bool,
    current_fn_returns_result: bool,
    
    /// Stack of "error expected" contexts
    error_handled_depth: u32,
    
    pub annotations: HashMap<usize, ErrorAnnotation>,
}

impl ErrorAnalyzer {
    pub fn new() -> Self {
        Self {
            current_fn_returns_option: false,
            current_fn_returns_result: false,
            error_handled_depth: 0,
            annotations: HashMap::new(),
        }
    }

    /// Analyze a function's return type to understand ? behavior
    fn set_return_type(&mut self, ty: &str) {
        self.current_fn_returns_option = ty.contains("Option");
        self.current_fn_returns_result = ty.contains("Result");
    }

    /// Detect ? operator usage
    fn analyze_question_mark(&mut self, expr_node_id: usize, fn_node_id: usize) {
        let handling = if self.current_fn_returns_option || self.current_fn_returns_result {
            ErrorHandling::Propagated
        } else {
            ErrorHandling::Propagated  // Still propagated to caller
        };

        self.annotations.insert(expr_node_id, ErrorAnnotation {
            handling,
            propagates_to: Some(fn_node_id),
            confidence: 2,  // proven from syntax
        });
    }

    /// Analyze entire function for error handling patterns
    pub fn analyze_function(&mut self, func_name: &str, return_type: &str, body: &str) {
        let fn_node_id = 0; // Simplified
        
        self.set_return_type(return_type);
        
        // Look for ? operator
        if body.contains('?') {
            self.analyze_question_mark(1, fn_node_id);
        }
        
        // Look for .unwrap() calls
        if body.contains(".unwrap()") {
            self.annotations.insert(2, ErrorAnnotation {
                handling: ErrorHandling::Unwrapped,
                propagates_to: None,
                confidence: 2,
            });
        }
        
        // Look for .expect() calls
        if body.contains(".expect(") {
            self.annotations.insert(3, ErrorAnnotation {
                handling: ErrorHandling::Expected,
                propagates_to: None,
                confidence: 2,
            });
        }
    }
}

/// Analyze error handling across an entire program
pub fn analyze_errors(source: &str) -> HashMap<usize, ErrorAnnotation> {
    let mut analyzer = ErrorAnalyzer::new();
    
    // Simplified: analyze whole source as one unit
    // Real impl would parse into functions first
    analyzer.analyze_function("main", "()", source);
    
    analyzer.annotations
}
