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
use pesti_structural::lexer::tokenize_rust;
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
    /// Weight for raw text embeddings (0.0..1.0)
    pub text_weight: f32,
    /// Weight for structural embeddings (0.0..1.0)
    pub struct_weight: f32,
}

impl Default for StructuralConfig {
    fn default() -> Self {
        Self {
            grammar_path: "crates/pesti-structural/grammars/rust.peg".to_string(),
            text_weight: 0.5,
            struct_weight: 0.5,
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
    pub fn extract_features(&mut self, source: &str) -> Result<Vec<EmbeddingFeature>> {
        let grammar = self.load_grammar()?;

        // Tokenize with the Rust lexer
        let (tokens, _errors) = tokenize_rust(source);

        // Parse with the PEG evaluator
        let evaluator = PegEvaluator::new(grammar, &tokens);
        let result: ParseResult = evaluator.parse();

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
        walk_for_features(&tree, &grammar_clone, &mut features);

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
    out: &mut Vec<EmbeddingFeature>,
) {
    // Check if this node's rule has an annotation
    let rule_name = node.label.strip_prefix("capture:").unwrap_or(&node.label);
    if let Some(ann) = find_annotation(grammar, rule_name) {
        let label = extract_label(node, &ann.embed_fields);
        out.push(EmbeddingFeature {
            role: ann.role.clone(),
            class: ann.similarity_class.clone().unwrap_or_default(),
            label,
        });
    }

    // Recurse into children and captures
    for child in &node.children {
        walk_for_features(child, grammar, out);
    }
    for capture in node.captures.values() {
        walk_for_features(capture, grammar, out);
    }
}

/// Find a rule annotation by nonterminal name.
fn find_annotation<'a>(grammar: &'a StructuralGrammar, name: &str) -> Option<&'a RuleAnnotation> {
    grammar.annotations.get(&Nonterminal(name.to_string()))
}

/// Extract a label from named captures specified in the annotation.
fn extract_label(node: &TreeNode, fields: &[String]) -> String {
    let mut parts = Vec::new();
    for field in fields {
        let text = find_capture_text(node, field);
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
fn find_capture_text(node: &TreeNode, name: &str) -> Option<String> {
    // Check direct captures first
    if let Some(captured) = node.captures.get(name) {
        return Some(collect_text(captured));
    }
    // Search children
    for child in &node.children {
        if let Some(text) = find_capture_text(child, name) {
            return Some(text);
        }
    }
    // Search nested captures
    for capture in node.captures.values() {
        if let Some(text) = find_capture_text(capture, name) {
            return Some(text);
        }
    }
    None
}

/// Collect all text content from a subtree.
fn collect_text(node: &TreeNode) -> String {
    if node.children.is_empty() && node.captures.is_empty() {
        // Terminal node — the label is the token kind or text
        node.label.clone()
    } else {
        let mut parts = Vec::new();
        for child in &node.children {
            parts.push(collect_text(child));
        }
        for capture in node.captures.values() {
            parts.push(collect_text(capture));
        }
        parts.join("")
    }
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

    #[test]
    fn test_extract_features_simple_function() {
        let grammar_path = "crates/pesti-structural/grammars/rust.peg";
        if !Path::new(grammar_path).exists() {
            eprintln!("Grammar file not found, skipping integration test");
            return;
        }

        let config = StructuralConfig {
            grammar_path: grammar_path.to_string(),
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
        let grammar_path = "crates/pesti-structural/grammars/rust.peg";
        if !Path::new(grammar_path).exists() {
            eprintln!("Grammar file not found, skipping integration test");
            return;
        }

        let config = StructuralConfig {
            grammar_path: grammar_path.to_string(),
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
}
