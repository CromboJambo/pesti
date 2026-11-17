//! Structure-aware embedding pipeline using pesti-structural's PEG grammar parser.
//!
//! Bridges the annotated parse tree from `pesti-structural` into the embedding
//! pipeline. The approach:
//!
//! 1. Parse source code with a PEG grammar → annotated tree with roles/captures
//! 2. Extract `EmbeddingFeature` entries (role, class, label) from annotations
//! 3. Format features as a text representation
//! 4. Embed both raw text and structural repr separately
//! 5. Combine with weighted average for structure-aware embeddings

use std::path::Path;

use anyhow::{Result, anyhow};
use pesti_structural::evaluator::{ParseResult, PegEvaluator};
use pesti_structural::grammar::{Nonterminal, RuleAnnotation, StructuralGrammar};
use pesti_structural::lexer::tokenize_for_language;
use pesti_structural::parser::parse_peg_file;
use pesti_structural::tree::TreeNode;

/// A single structural feature extracted from an annotated parse tree.
#[derive(Debug, Clone)]
pub struct EmbeddingFeature {
    /// Semantic role (e.g., "definition", "declaration", "implementation")
    pub role: String,
    /// Similarity class — nodes in the same class should embed near each other
    pub class: String,
    /// Human-readable label derived from named captures
    pub label: String,
}

/// Configuration for structure-aware embedding.
#[derive(Debug, Clone)]
pub struct StructuralConfig {
    /// Path to the PEG grammar file (e.g., "grammars/rust.peg")
    pub grammar_path: String,
    /// Programming language for lexing ("rust", "python"). Derived from
    /// grammar_path if not explicitly set.
    pub language: Option<String>,
    /// Weight for raw text embeddings (0.0..1.0)
    pub text_weight: f32,
    /// Weight for structural embeddings (0.0..1.0)
    pub struct_weight: f32,
}

impl Default for StructuralConfig {
    fn default() -> Self {
        Self {
            grammar_path: "crates/pesti-structural/grammars/rust.peg".to_string(),
            language: None,
            text_weight: 0.5,
            struct_weight: 0.5,
        }
    }
}

impl StructuralConfig {
    /// Determine the language to use for lexing.
    /// Uses the explicit `language` field if set, otherwise derives it from
    /// the grammar filename (e.g., "python.peg" → "python").
    pub fn effective_language(&self) -> &str {
        if let Some(lang) = &self.language {
            return lang;
        }
        // Derive from grammar filename.
        let path = self.grammar_path.as_str();
        if path.contains("python") {
            "python"
        } else if path.contains("typescript") || path.contains(".ts") {
            "typescript"
        } else {
            "rust"
        }
    }
}

/// Structure-aware embedding engine.
///
/// Combines semantic embeddings (from raw text) with structural embeddings
/// (from PEG grammar annotations) to produce representations that capture
/// both meaning and syntactic form.
pub struct StructuralEmbedder {
    config: StructuralConfig,
    /// Cached parsed grammar (loaded lazily on first use)
    grammar: Option<StructuralGrammar>,
}

impl StructuralEmbedder {
    /// Create a new structural embedder with the given configuration.
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(config: StructuralConfig) -> Self {
        Self {
            config,
            grammar: None,
        }
    }

    /// Load and cache the PEG grammar from disk.
    fn load_grammar(&mut self) -> Result<&StructuralGrammar> {
        if self.grammar.is_none() {
            let path = Path::new(&self.config.grammar_path);
            let grammar = parse_peg_file(path).map_err(|e| {
                anyhow!(
                    "Failed to load grammar from '{}': {}",
                    self.config.grammar_path,
                    e
                )
            })?;
            self.grammar = Some(grammar);
        }
        Ok(self.grammar.as_ref().unwrap())
    }

    /// Parse source code and extract structural features.
    ///
    /// Returns a list of `EmbeddingFeature` entries derived from the annotated
    /// parse tree's rule annotations (role, similarity_class, embed_fields).
    ///
    /// Uses error recovery: if the grammar doesn't cover every construct in
    /// the source, we still extract features from the successfully parsed
    /// portions rather than failing the entire file.
    pub fn extract_features(&mut self, source: &str) -> Result<Vec<EmbeddingFeature>> {
        // Get the language first (immutable read of config).
        let language = self.config.effective_language().to_string();

        let grammar = self.load_grammar()?;

        // Tokenize with the appropriate language lexer.
        let (tokens, _errors) = tokenize_for_language(source, &language);

        if tokens.is_empty() {
            return Ok(Vec::new());
        }

        // Parse with the PEG evaluator using error recovery.
        // This returns partial trees for files with constructs our grammar
        // doesn't cover yet, instead of failing entirely.
        let evaluator = PegEvaluator::new(grammar, &tokens);
        let result: ParseResult = evaluator.parse_with_recovery();

        if result.tree.is_none() {
            return Err(anyhow!(
                "Parse failed: {}",
                result
                    .errors
                    .first()
                    .map(|e| e.message.clone())
                    .unwrap_or_else(|| "unknown error".to_string())
            ));
        }

        let tree = result.tree.unwrap();

        // Walk the tree and extract features from annotated rules.
        // Clone the grammar reference to avoid borrow conflicts with self.
        let grammar_clone = grammar.clone();
        let mut features = Vec::new();
        walk_for_features(&tree, &grammar_clone, source, &mut features);

        Ok(features)
    }

    /// Format structural features as an embeddable text representation.
    ///
    /// The format is designed to be tokenizable and semantically meaningful:
    /// ```text
    /// [role=definition class=declaration] struct Point
    /// [role=implementation class=declaration] impl Point
    /// [role=expression class=construction] Point { x, y }
    /// ```
    pub fn format_structural_repr(features: &[EmbeddingFeature]) -> String {
        features
            .iter()
            .map(|f| format!("[role={} class={}] {}", f.role, f.class, f.label))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Compute similarity groups from features.
    ///
    /// Returns a map from similarity class to list of feature indices.
    /// Nodes in the same group should embed near each other in vector space.
    pub fn similarity_groups(features: &[EmbeddingFeature]) -> Vec<(String, Vec<usize>)> {
        use std::collections::BTreeMap;
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (i, f) in features.iter().enumerate() {
            if !f.class.is_empty() {
                groups.entry(f.class.clone()).or_default().push(i);
            }
        }
        groups.into_iter().collect()
    }
}

/// Recursively walk the parse tree, extracting features from annotated nodes.
fn walk_for_features(
    node: &TreeNode,
    grammar: &StructuralGrammar,
    source: &str,
    out: &mut Vec<EmbeddingFeature>,
) {
    // Check if this node's rule has an annotation
    let rule_name = node.label.strip_prefix("capture:").unwrap_or(&node.label);
    if let Some(ann) = find_annotation(grammar, rule_name) {
        let label = extract_label(node, &ann.embed_fields, source);
        out.push(EmbeddingFeature {
            role: ann.role.clone(),
            class: ann.similarity_class.clone().unwrap_or_default(),
            label,
        });
    }

    // Recurse into children and captures
    for child in &node.children {
        walk_for_features(child, grammar, source, out);
    }
    for capture in node.captures.values() {
        walk_for_features(capture, grammar, source, out);
    }
}

/// Find a rule annotation by nonterminal name.
fn find_annotation<'a>(grammar: &'a StructuralGrammar, name: &str) -> Option<&'a RuleAnnotation> {
    grammar.annotations.get(&Nonterminal(name.to_string()))
}

/// Extract a label from named captures specified in the annotation.
fn extract_label(node: &TreeNode, fields: &[String], source: &str) -> String {
    let mut parts = Vec::new();
    for field in fields {
        let text = find_capture_text(node, field, source);
        if let Some(t) = text.filter(|s| !s.is_empty()) {
            parts.push(t);
        }
    }
    if parts.is_empty() {
        "<no captures>".to_string()
    } else {
        parts.join(":")
    }
}

/// Recursively search for a named capture under this node.
fn find_capture_text(node: &TreeNode, name: &str, source: &str) -> Option<String> {
    // Check direct captures first
    if let Some(captured) = node.captures.get(name) {
        return Some(collect_text(captured, source));
    }
    // Search children
    for child in &node.children {
        if let Some(text) = find_capture_text(child, name, source) {
            return Some(text);
        }
    }
    // Search nested captures
    for capture in node.captures.values() {
        if let Some(text) = find_capture_text(capture, name, source) {
            return Some(text);
        }
    }
    None
}

/// Collect all text content from a subtree.
fn collect_text(node: &TreeNode, source: &str) -> String {
    // Use the span to extract actual source text — this works for both
    // terminal nodes (single token) and internal nodes (compound spans).
    node.span.text(source).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_structural_repr() {
        let features = vec![
            EmbeddingFeature {
                role: "definition".to_string(),
                class: "declaration".to_string(),
                label: "struct Point".to_string(),
            },
            EmbeddingFeature {
                role: "implementation".to_string(),
                class: "declaration".to_string(),
                label: "impl Point".to_string(),
            },
        ];

        let repr = StructuralEmbedder::format_structural_repr(&features);
        assert!(repr.contains("[role=definition class=declaration] struct Point"));
        assert!(repr.contains("[role=implementation class=declaration] impl Point"));
    }

    #[test]
    fn test_similarity_groups() {
        let features = vec![
            EmbeddingFeature {
                role: "definition".to_string(),
                class: "declaration".to_string(),
                label: "struct Point".to_string(),
            },
            EmbeddingFeature {
                role: "implementation".to_string(),
                class: "declaration".to_string(),
                label: "impl Point".to_string(),
            },
            EmbeddingFeature {
                role: "expression".to_string(),
                class: "construction".to_string(),
                label: "Point { x, y }".to_string(),
            },
        ];

        let groups = StructuralEmbedder::similarity_groups(&features);
        assert_eq!(groups.len(), 2);

        // Check declaration group
        let decl_group = groups.iter().find(|(c, _)| c == "declaration").unwrap();
        assert_eq!(decl_group.1, vec![0, 1]);

        // Check construction group
        let constr_group = groups.iter().find(|(c, _)| c == "construction").unwrap();
        assert_eq!(constr_group.1, vec![2]);
    }

    #[test]
    fn test_similarity_groups_empty_class() {
        let features = vec![EmbeddingFeature {
            role: "definition".to_string(),
            class: "".to_string(),
            label: "fn main".to_string(),
        }];

        let groups = StructuralEmbedder::similarity_groups(&features);
        assert!(groups.is_empty());
    }

    #[test]
    fn test_structural_config_default() {
        let config = StructuralConfig::default();
        assert_eq!(config.text_weight, 0.5);
        assert_eq!(config.struct_weight, 0.5);
        assert!(config.grammar_path.contains("rust.peg"));
    }

    /// Resolve a grammar path relative to the workspace root.
    fn workspace_grammar_path(grammar: &str) -> String {
        // CARGO_MANIFEST_DIR is crates/pesti-embeddings; go up two levels to workspace root.
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
        workspace_root
            .join("crates")
            .join("pesti-structural")
            .join("grammars")
            .join(grammar)
            .to_string_lossy()
            .to_string()
    }

    #[test]
    fn test_extract_features_simple_function() {
        let grammar_path = workspace_grammar_path("rust.peg");
        if !Path::new(&grammar_path).exists() {
            eprintln!("Grammar file not found at {}, skipping", grammar_path);
            return;
        }

        let config = StructuralConfig {
            grammar_path,
            language: None,
            ..Default::default()
        };
        let mut embedder = StructuralEmbedder::new(config);

        let source = r#"fn add(a: i32, b: i32) -> i32 {
    return a + b;
}"#;

        let features = embedder.extract_features(source).unwrap();
        // Should extract at least one feature (the function definition)
        assert!(!features.is_empty());

        // Check that at least one feature has a meaningful role
        let has_definition = features.iter().any(|f| f.role == "definition");
        assert!(
            has_definition,
            "Expected at least one 'definition' role feature"
        );
    }

    #[test]
    fn test_extract_features_struct() {
        let grammar_path = workspace_grammar_path("rust.peg");
        if !Path::new(&grammar_path).exists() {
            eprintln!("Grammar file not found at {}, skipping", grammar_path);
            return;
        }

        let config = StructuralConfig {
            grammar_path,
            language: None,
            ..Default::default()
        };
        let mut embedder = StructuralEmbedder::new(config);

        let source = r#"pub struct Point {
    pub x: f64,
    pub y: f64,
}"#;

        let features = embedder.extract_features(source).unwrap();
        assert!(!features.is_empty());

        // Check that we extracted a struct definition
        let has_struct = features.iter().any(|f| f.label.contains("Point"));
        assert!(has_struct, "Expected to find 'Point' in feature labels");
    }

    #[test]
    fn test_extract_features_python_simple() {
        let grammar_path = workspace_grammar_path("python.peg");
        if !Path::new(&grammar_path).exists() {
            eprintln!("Python grammar file not found at {}, skipping", grammar_path);
            return;
        }

        let config = StructuralConfig {
            grammar_path,
            language: Some("python".to_string()),
            ..Default::default()
        };
        let mut embedder = StructuralEmbedder::new(config);

        let source = r#"def add(a: int, b: int) -> int:
    return a + b

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
"#;

        match embedder.extract_features(source) {
            Ok(features) => {
                assert!(
                    !features.is_empty(),
                    "Expected at least one feature from Python source"
                );
                // Check we found the function and class definitions
                let has_function = features.iter().any(|f| f.label.contains("add"));
                let has_class = features.iter().any(|f| f.label.contains("Point"));
                assert!(has_function, "Expected to find 'add' function in features");
                assert!(has_class, "Expected to find 'Point' class in features");
            }
            Err(e) => {
                panic!("Failed to parse Python source: {}", e);
            }
        }
    }

    #[test]
    fn test_extract_features_python_real_file() {
        let grammar_path = workspace_grammar_path("python.peg");
        if !Path::new(&grammar_path).exists() {
            eprintln!("Python grammar file not found at {}, skipping", grammar_path);
            return;
        }

        // Test with real Python files from the project.
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();

        let test_files = [
            "conformance-corpus/triangulate.py",
            "conformance-corpus/dequant_test.py",
            "references/reference_constants.py",
        ];

        let config = StructuralConfig {
            grammar_path,
            language: Some("python".to_string()),
            ..Default::default()
        };
        let mut embedder = StructuralEmbedder::new(config);

        for file in &test_files {
            let full_path = workspace_root.join(file);
            if !full_path.exists() {
                continue;
            }
            let content = std::fs::read_to_string(&full_path).unwrap();
            match embedder.extract_features(&content) {
                Ok(features) => {
                    println!("  {}: {} features extracted", file, features.len());
                }
                Err(e) => {
                    eprintln!("  {}: ERROR: {}", file, e);
                }
            }
        }
    }

    #[test]
    fn test_language_derivation_from_path() {
        let config = StructuralConfig {
            grammar_path: "crates/pesti-structural/grammars/python.peg".to_string(),
            language: None,
            ..Default::default()
        };
        assert_eq!(config.effective_language(), "python");

        let config = StructuralConfig {
            grammar_path: "crates/pesti-structural/grammars/rust.peg".to_string(),
            language: None,
            ..Default::default()
        };
        assert_eq!(config.effective_language(), "rust");

        // Explicit language overrides path derivation
        let config = StructuralConfig {
            grammar_path: "crates/pesti-structural/grammars/rust.peg".to_string(),
            language: Some("python".to_string()),
            ..Default::default()
        };
        assert_eq!(config.effective_language(), "python");
    }

    /// Validate that similar code produces similar structural features,
    /// and different code produces distinct features.
    #[test]
    fn test_structural_similarity_validation() {
        let grammar_path = workspace_grammar_path("python.peg");
        if !Path::new(&grammar_path).exists() {
            eprintln!("Python grammar file not found at {}, skipping", grammar_path);
            return;
        }

        let config = StructuralConfig {
            grammar_path,
            language: Some("python".to_string()),
            ..Default::default()
        };

        // Two similar functions (same structure, different names)
        let func_a = r#"def calculate_sum(a: int, b: int) -> int:
    return a + b"#;
        let func_b = r#"def compute_total(x: int, y: int) -> int:
    return x + y"#;

        // A very different construct (class with method)
        let class_c = r#"class DataProcessor:
    def __init__(self, threshold: float):
        self.threshold = threshold

    def process(self, value: float) -> bool:
        return value > self.threshold"#;

        let mut embedder_a = StructuralEmbedder::new(config.clone());
        let mut embedder_b = StructuralEmbedder::new(config.clone());
        let mut embedder_c = StructuralEmbedder::new(config);

        let features_a = embedder_a.extract_features(func_a).unwrap();
        let features_b = embedder_b.extract_features(func_b).unwrap();
        let features_c = embedder_c.extract_features(class_c).unwrap();

        // All should produce features
        assert!(!features_a.is_empty(), "func_a produced no features");
        assert!(!features_b.is_empty(), "func_b produced no features");
        assert!(!features_c.is_empty(), "class_c produced no features");

        // Similar functions should share structural characteristics:
        // - Both have 'definition' roles
        // - Both should have similar feature counts (within 2x)
        let roles_a: Vec<&str> = features_a.iter().map(|f| f.role.as_str()).collect();
        let roles_b: Vec<&str> = features_b.iter().map(|f| f.role.as_str()).collect();

        assert!(roles_a.contains(&"definition"), "func_a missing 'definition' role");
        assert!(roles_b.contains(&"definition"), "func_b missing 'definition' role");

        // Feature counts should be in the same ballpark for similar structures
        let count_ratio = features_a.len() as f32 / features_b.len().max(1) as f32;
        assert!(
            (0.5..=2.0).contains(&count_ratio),
            "Similar functions have very different feature counts: {} vs {}",
            features_a.len(),
            features_b.len()
        );

        // Different construct should have different characteristics:
        // - The class should produce more features (has __init__ + process method)
        assert!(
            features_c.len() > features_a.len(),
            "Class with methods ({}) should have more features than single function ({})",
            features_c.len(),
            features_a.len()
        );

        // The class should contain multiple definitions
        let def_count_c = features_c.iter().filter(|f| f.role == "definition").count();
        assert!(
            def_count_c >= 2,
            "Class with methods should have at least 2 definitions, got {}",
            def_count_c
        );

        // Labels should be distinct (different names)
        let labels_a: Vec<&str> = features_a.iter().map(|f| f.label.as_str()).collect();
        let labels_b: Vec<&str> = features_b.iter().map(|f| f.label.as_str()).collect();
        assert!(
            labels_a.iter().any(|l| l.contains("calculate_sum")),
            "func_a label should contain 'calculate_sum'"
        );
        assert!(
            labels_b.iter().any(|l| l.contains("compute_total")),
            "func_b label should contain 'compute_total'"
        );
    }
}
