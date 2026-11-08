//! Complexity analysis for Rust source code.
//! Lightweight string-based scanner to compute cyclomatic complexity,
//! nesting depth, and control flow density metrics.

use std::collections::HashMap;

/// Complexity metrics for a function or code region
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComplexityMetrics {
    /// Cyclomatic complexity: number of independent paths through the code
    pub cyclomatic: u32,
    /// Maximum nesting depth observed
    pub max_depth: u32,
    /// Count of control flow statements (if, while, for, match, return, break, continue)
    pub control_flow_count: u32,
}

/// Annotation attached to AST nodes by the complexity analyzer
#[derive(Debug, Clone)]
pub struct ComplexityAnnotation {
    pub metrics: ComplexityMetrics,
    /// Confidence: 0=unknown, 1=inferred, 2=proven from syntax
    pub confidence: u8,
}

/// Minimal complexity analyzer — tracks nesting and control flow through the AST
pub struct ComplexityAnalyzer {
    /// Current scope's variable -> ownership mode
    scopes: Vec<HashMap<String, ComplexityMetrics>>,
    /// Annotations to emit (node_id -> annotation)
    pub annotations: HashMap<usize, ComplexityAnnotation>,
}

impl ComplexityAnalyzer {
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

    /// Record a complexity annotation for a code region
    fn annotate(&mut self, node_id: usize, metrics: ComplexityMetrics) {
        self.annotations.insert(
            node_id,
            ComplexityAnnotation {
                metrics,
                confidence: 2, // proven from syntax
            },
        );
    }

    /// Analyze a function for complexity patterns (string-based scanning)
    pub fn analyze_function(&mut self, func: &str) -> ComplexityMetrics {
        let node_id = 0; // Simplified - real impl would use actual node IDs

        let metrics = compute_complexity(func);
        self.annotate(node_id, metrics);
        metrics
    }
}

/// Compute complexity metrics for a Rust source string.
pub fn analyze_complexity(source: &str) -> HashMap<usize, ComplexityAnnotation> {
    let mut analyzer = ComplexityAnalyzer::new();

    // Simplified: treat whole source as one function body for now.
    // Real impl would parse into functions first.
    analyzer.enter_scope();

    let metrics = compute_complexity(source);
    analyzer.annotate(0, metrics);

    analyzer.exit_scope();
    analyzer.annotations
}

/// Compute complexity metrics from source text using lightweight scanning.
fn compute_complexity(source: &str) -> ComplexityMetrics {
    let mut nesting_depth = 0;
    let mut max_depth = 0;
    let mut control_flow_count = 0;

    for line in source.lines() {
        let trimmed = line.trim();

        // Count opening braces as nesting depth increases
        if trimmed.ends_with('{') {
            nesting_depth += 1;
            if nesting_depth > max_depth {
                max_depth = nesting_depth;
            }
        }

        // Count closing braces as nesting depth decreases
        if trimmed == "}" || trimmed.starts_with("} else") || trimmed == "};" {
            if nesting_depth > 0 {
                nesting_depth -= 1;
            }
        }

        // Count control flow statements
        if trimmed.starts_with("if ") || trimmed.contains(" else if ") {
            control_flow_count += 1;
        }
        if trimmed.starts_with("while ") {
            control_flow_count += 1;
        }
        if trimmed.starts_with("for ") {
            control_flow_count += 1;
        }
        if trimmed.starts_with("match ") {
            control_flow_count += 1;
        }
        if trimmed == "return;" || trimmed.starts_with("return ") {
            control_flow_count += 1;
        }
        if trimmed == "break;" || trimmed.starts_with("break ") {
            control_flow_count += 1;
        }
        if trimmed == "continue;" || trimmed.starts_with("continue ") {
            control_flow_count += 1;
        }
    }

    let cyclomatic = control_flow_count + 1;
    ComplexityMetrics {
        cyclomatic,
        max_depth,
        control_flow_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function() {
        let source = r#"
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}
"#;
        let annotations = analyze_complexity(source);
        assert_eq!(annotations.len(), 1);
        let annotation = &annotations[&0];
        // One control flow (return), cyclomatic = 2, max depth = 1
        assert_eq!(annotation.metrics.cyclomatic, 2);
        assert_eq!(annotation.metrics.max_depth, 1);
        assert_eq!(annotation.metrics.control_flow_count, 1);
    }

    #[test]
    fn test_if_else_complexity() {
        let source = r#"
fn classify(x: i32) -> &'static str {
    if x < 0 {
        return "negative";
    } else if x == 0 {
        return "zero";
    } else {
        return "positive";
    }
}
"#;
        let annotations = analyze_complexity(source);
        assert_eq!(annotations.len(), 1);
        let annotation = &annotations[&0];
        // 2 if statements + 3 returns = 5 control flow, cyclomatic = 6
        assert_eq!(annotation.metrics.cyclomatic, 6);
    }

    #[test]
    fn test_loop_complexity() {
        let source = r#"
fn process(items: &[i32]) {
    for item in items {
        if *item > 10 {
            continue;
        }
        println!("{}", item);
    }
}
"#;
        let annotations = analyze_complexity(source);
        assert_eq!(annotations.len(), 1);
        let annotation = &annotations[&0];
        // for + if + continue = 3 control flow, cyclomatic = 4
        assert_eq!(annotation.metrics.cyclomatic, 4);
    }

    #[test]
    fn test_match_complexity() {
        let source = r#"
fn handle(msg: &str) {
    match msg {
        "hello" => println!("hi"),
        "bye" => println!("see you"),
        _ => println!("unknown"),
    }
}
"#;
        let annotations = analyze_complexity(source);
        assert_eq!(annotations.len(), 1);
        let annotation = &annotations[&0];
        // 1 match statement, cyclomatic = 2
        assert_eq!(annotation.metrics.cyclomatic, 2);
    }

    #[test]
    fn test_nesting_depth() {
        let source = r#"
fn deeply_nested(x: i32) {
    if x > 0 {
        for i in 0..x {
            if i % 2 == 0 {
                println!("{}", i);
            }
        }
    }
}
"#;
        let annotations = analyze_complexity(source);
        assert_eq!(annotations.len(), 1);
        let annotation = &annotations[&0];
        // Max nesting depth: fn body(1) > if(2) > for(3) > if(4)
        assert_eq!(annotation.metrics.max_depth, 4);
    }
}