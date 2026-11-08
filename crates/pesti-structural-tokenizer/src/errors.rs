//! Structured error types for pesti-structural-tokenizer with context and recovery support.
//!
//! This module provides improved error handling for crabjar's JEV decision-making system,
//! enabling distinction between syntax errors in input code, tokenizer internal errors,
//! and analyzer-specific failures.

use std::collections::HashMap;

/// Error severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Fatal error that prevents any further analysis
    Fatal,
    /// Recoverable error — partial results may still be useful
    Recoverable,
    /// Warning — analysis completed but with reduced confidence
    Warning,
}

/// Context information about where an error occurred
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Byte position in source
    pub pos: usize,
    /// Line number (1-indexed)
    pub line: usize,
    /// Column number (1-indexed)
    pub col: usize,
    /// Source snippet around the error
    pub snippet: String,
}

impl ErrorContext {
    /// Create context from source position
    pub fn new(source: &str, pos: usize) -> Self {
        // Calculate line and column
        let before = &source[..pos.min(source.len())];
        let lines = before.lines().count();
        let last_newline = before.rfind('\n').map(|i| i + 1).unwrap_or(0);
        let col = pos - last_newline + 1;

        // Extract snippet (20 chars before, 40 after)
        let start = if pos >= 20 { pos - 20 } else { 0 };
        let end = (pos + 40).min(source.len());
        let snippet = source[start..end].to_string();

        Self {
            pos,
            line: lines + 1,
            col,
            snippet,
        }
    }
}

/// Structured error type with context and severity
#[derive(Debug, Clone)]
pub struct TokenizerError {
    /// Error kind
    pub kind: ErrorKind,
    /// Human-readable message
    pub message: String,
    /// Context about where the error occurred
    pub context: Option<ErrorContext>,
    /// Severity level
    pub severity: Severity,
}

/// Categories of tokenizer errors
#[derive(Debug, Clone)]
pub enum ErrorKind {
    /// Syntax error in input code (e.g., unclosed brace)
    Syntax { detail: String },
    /// Internal tokenizer error (bug or unexpected state)
    Internal { detail: String },
    /// Budget exceeded during tokenization
    BudgetExceeded { tokens_emitted: usize, budget: usize },
    /// Semantic analyzer failure
    AnalyzerError { analyzer: String, detail: String },
    /// Input too large for available memory
    Memory { required_bytes: usize },
}

impl TokenizerError {
    /// Create a syntax error with context
    pub fn syntax(source: &str, pos: usize, detail: impl Into<String>) -> Self {
        let detail_str = detail.into();
        let context = ErrorContext::new(source, pos);
        Self {
            kind: ErrorKind::Syntax {
                detail: detail_str.clone(),
            },
            message: format!("Syntax error at line {}: {}", context.line, detail_str),
            context: Some(context),
            severity: Severity::Fatal,
        }
    }

    /// Create an internal error
    pub fn internal(detail: impl Into<String>) -> Self {
        let detail_str = detail.into();
        Self {
            kind: ErrorKind::Internal {
                detail: detail_str.clone(),
            },
            message: format!("Internal tokenizer error: {}", detail_str),
            context: None,
            severity: Severity::Fatal,
        }
    }

    /// Create a budget exceeded warning
    pub fn budget_exceeded(tokens_emitted: usize, budget: usize) -> Self {
        Self {
            kind: ErrorKind::BudgetExceeded {
                tokens_emitted,
                budget,
            },
            message: format!(
                "Tokenization budget exceeded: {} tokens emitted (budget: {})",
                tokens_emitted, budget
            ),
            context: None,
            severity: Severity::Warning,
        }
    }

    /// Create an analyzer error
    pub fn analyzer_error(analyzer: &str, detail: impl Into<String>) -> Self {
        let detail_str = detail.into();
        Self {
            kind: ErrorKind::AnalyzerError {
                analyzer: analyzer.to_string(),
                detail: detail_str.clone(),
            },
            message: format!("Analyzer '{}' error: {}", analyzer, detail_str),
            context: None,
            severity: Severity::Recoverable,
        }
    }

    /// Check if this error is fatal
    pub fn is_fatal(&self) -> bool {
        self.severity == Severity::Fatal
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(self.severity, Severity::Recoverable | Severity::Warning)
    }
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.severity, self.message)?;
        if let Some(ctx) = &self.context {
            write!(f, " (line {}, col {})", ctx.line, ctx.col)?;
        }
        Ok(())
    }
}

impl std::error::Error for TokenizerError {}

/// Result type with error context
pub type TokenizerResult<T> = Result<T, TokenizerError>;

/// Recovery mode configuration for best-effort analysis
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Continue after syntax errors (emit partial results)
    pub continue_on_syntax_error: bool,
    /// Maximum number of errors to collect before giving up
    pub max_errors: usize,
    /// Collect warnings even in non-recovery mode
    pub collect_warnings: bool,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            continue_on_syntax_error: false,
            max_errors: 10,
            collect_warnings: true,
        }
    }
}

/// Analysis result with errors and warnings collected during recovery mode
#[derive(Debug)]
pub struct AnalysisResult<T> {
    /// The analysis result (may be partial in recovery mode)
    pub value: Option<T>,
    /// Errors encountered during analysis
    pub errors: Vec<TokenizerError>,
    /// Warnings encountered during analysis
    pub warnings: Vec<TokenizerError>,
}

impl<T> AnalysisResult<T> {
    /// Create a successful result
    pub fn ok(value: T) -> Self {
        Self {
            value: Some(value),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Create a result with errors (recovery mode)
    pub fn with_errors(errors: Vec<TokenizerError>) -> Self {
        Self {
            value: None,
            errors,
            warnings: Vec::new(),
        }
    }

    /// Add an error to this result
    pub fn add_error(&mut self, error: TokenizerError) {
        if !error.is_fatal() || self.value.is_some() {
            self.errors.push(error);
        }
    }

    /// Add a warning to this result
    pub fn add_warning(&mut self, warning: TokenizerError) {
        self.warnings.push(warning);
    }

    /// Check if any fatal errors occurred
    pub fn has_fatal_errors(&self) -> bool {
        self.errors.iter().any(|e| e.is_fatal())
    }

    /// Convert to a Result (errors become Err, warnings are ignored)
    pub fn into_result(self) -> TokenizerResult<T> {
        match self.value {
            Some(value) => Ok(value),
            None if !self.errors.is_empty() => Err(self.errors[0].clone()),
            None => Err(TokenizerError::internal("Analysis failed with no specific error")),
        }
    }
}

/// Compatibility wrapper for the legacy TokenizeError type
#[derive(Debug)]
pub enum LegacyTokenizeError {
    Parse { pos: usize, msg: String },
}

impl std::fmt::Display for LegacyTokenizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LegacyTokenizeError::Parse { pos, msg } => {
                write!(f, "Parse error at position {}: {}", pos, msg)
            }
        }
    }
}

impl std::error::Error for LegacyTokenizeError {}

/// Convert legacy error to new structured error
pub fn legacy_to_structured(source: &str, err: LegacyTokenizeError) -> TokenizerError {
    match err {
        LegacyTokenizeError::Parse { pos, msg } => TokenizerError::syntax(source, pos, msg),
    }
}

/// Convert new structured error to legacy format (for backward compatibility)
pub fn structured_to_legacy(err: &TokenizerError) -> Option<LegacyTokenizeError> {
    match &err.kind {
        ErrorKind::Syntax { detail } => {
            if let Some(ctx) = &err.context {
                Some(LegacyTokenizeError::Parse {
                    pos: ctx.pos,
                    msg: detail.clone(),
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntax_error_with_context() {
        let source = "fn foo() {\n    if (x) {\n}\n";
        let err = TokenizerError::syntax(source, 20, "unclosed brace");
        assert!(err.is_fatal());
        assert_eq!(err.severity, Severity::Fatal);
        assert!(err.context.is_some());
        let ctx = err.context.as_ref().unwrap();
        assert_eq!(ctx.line, 3);
    }

    #[test]
    fn test_internal_error() {
        let err = TokenizerError::internal("unexpected state");
        assert!(err.is_fatal());
        assert!(err.context.is_none());
    }

    #[test]
    fn test_budget_exceeded_warning() {
        let err = TokenizerError::budget_exceeded(1500, 1000);
        assert!(err.is_recoverable());
        assert_eq!(err.severity, Severity::Warning);
    }

    #[test]
    fn test_analyzer_error() {
        let err = TokenizerError::analyzer_error("complexity", "failed to parse function");
        assert!(err.is_recoverable());
        assert_eq!(err.severity, Severity::Recoverable);
    }

    #[test]
    fn test_analysis_result_ok() {
        let result: AnalysisResult<Vec<i32>> = AnalysisResult::ok(vec![1, 2, 3]);
        assert!(result.value.is_some());
        assert!(!result.has_fatal_errors());
    }

    #[test]
    fn test_analysis_result_with_errors() {
        let mut result: AnalysisResult<Vec<i32>> = AnalysisResult::ok(vec![1, 2, 3]);
        result.add_error(TokenizerError::analyzer_error("ownership", "unknown pattern"));
        assert!(result.value.is_some()); // Still has value despite recoverable error
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_legacy_conversion() {
        let legacy = LegacyTokenizeError::Parse {
            pos: 42,
            msg: "unexpected token".to_string(),
        };
        let source = "fn foo() { unexpected }";
        let structured = legacy_to_structured(source, legacy);
        assert!(structured.is_fatal());

        // Round-trip back to legacy
        let back = structured_to_legacy(&structured);
        assert!(back.is_some());
    }
}