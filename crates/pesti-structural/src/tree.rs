//! Parse tree types with provenance tracking.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Source span: byte range in the original source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    /// Byte offset of the first character (inclusive).
    pub start: usize,
    /// Byte offset one past the last character (exclusive).
    pub end: usize,
}

impl Span {
    /// Create a new span.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Length of the span in bytes.
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    /// Whether the span is empty.
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Extract the matched text from the source.
    /// Safe for spans that don't align with UTF-8 character boundaries —
    /// adjusts indices to the nearest valid boundary.
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        let start = if self.start <= source.len() && source.is_char_boundary(self.start) {
            self.start
        } else {
            // Advance to the next valid char boundary
            (self.start.min(source.len())..=source.len())
                .find(|&i| source.is_char_boundary(i))
                .unwrap_or(source.len())
        };
        let end = if self.end <= source.len() && source.is_char_boundary(self.end) {
            self.end
        } else {
            // Advance to the next valid char boundary
            (self.end.min(source.len())..=source.len())
                .find(|&i| source.is_char_boundary(i))
                .unwrap_or(source.len())
        };
        if start >= end || start > source.len() || end > source.len() {
            ""
        } else {
            &source[start..end]
        }
    }

    /// Merge two spans into one covering both.
    pub fn merge(&self, other: &Span) -> Span {
        Span::new(self.start.min(other.start), self.end.max(other.end))
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}..{}]", self.start, self.end)
    }
}

/// A node in the parse tree. Each node records which rule matched it,
/// its span in the source, and its children (the subexpressions that matched).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    /// The nonterminal rule that produced this node.
    pub label: String,

    /// Source span covered by this node.
    pub span: Span,

    /// Child nodes (in order of appearance in the expression).
    pub children: Vec<TreeNode>,

    /// Named captures within this node, keyed by capture name.
    /// Each maps to the subtree that was captured.
    #[serde(default)]
    pub captures: std::collections::BTreeMap<String, TreeNode>,
}

impl TreeNode {
    /// Create a leaf node (terminal match).
    pub fn terminal(label: impl Into<String>, span: Span) -> Self {
        Self {
            label: label.into(),
            span,
            children: Vec::new(),
            captures: std::collections::BTreeMap::new(),
        }
    }

    /// Create an internal node with children.
    pub fn internal(label: impl Into<String>, span: Span, children: Vec<TreeNode>) -> Self {
        Self {
            label: label.into(),
            span,
            children,
            captures: std::collections::BTreeMap::new(),
        }
    }

    /// Add a named capture to this node.
    pub fn with_capture(mut self, name: impl Into<String>, node: TreeNode) -> Self {
        self.captures.insert(name.into(), node);
        self
    }

    /// Compute the depth of this subtree (leaves have depth 0).
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            0
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    /// Count the total number of nodes in this subtree.
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Format the tree as an indented string (for debugging).
    pub fn format_tree(&self, indent: usize, source: &str) -> String {
        let prefix = "  ".repeat(indent);
        let text = self.span.text(source);
        let display_text = if text.len() > 40 {
            format!("{}...", &text[..40])
        } else {
            text.to_string()
        };
        let mut out = format!(
            "{}{} {}
",
            prefix, self.label, display_text
        );
        for child in &self.children {
            out.push_str(&child.format_tree(indent + 1, source));
        }
        for (name, node) in &self.captures {
            out.push_str(&format!("{}[{}]\n", prefix, name));
            out.push_str(&node.format_tree(indent + 2, source));
        }
        out
    }

    /// Pretty-print the tree with source text.
    pub fn pretty(&self, source: &str) -> String {
        self.format_tree(0, source)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_basic() {
        let span = Span::new(5, 10);
        assert_eq!(span.len(), 5);
        assert!(!span.is_empty());
        assert_eq!(span.text("hello world"), " worl");
    }

    #[test]
    fn test_span_merge() {
        let a = Span::new(0, 5);
        let b = Span::new(3, 10);
        let merged = a.merge(&b);
        assert_eq!(merged, Span::new(0, 10));
    }

    #[test]
    fn test_tree_depth() {
        let leaf = TreeNode::terminal("Ident", Span::new(0, 3));
        assert_eq!(leaf.depth(), 0);

        let parent = TreeNode::internal("FuncDef", Span::new(0, 20), vec![leaf]);
        assert_eq!(parent.depth(), 1);
    }

    #[test]
    fn test_tree_node_count() {
        let leaf1 = TreeNode::terminal("a", Span::new(0, 1));
        let leaf2 = TreeNode::terminal("b", Span::new(1, 2));
        let parent = TreeNode::internal("Seq", Span::new(0, 2), vec![leaf1, leaf2]);
        assert_eq!(parent.node_count(), 3);
    }

    #[test]
    fn test_tree_with_capture() {
        let name_node = TreeNode::terminal("Ident", Span::new(3, 8));
        let node = TreeNode::internal("FuncDef", Span::new(0, 20), vec![name_node.clone()])
            .with_capture("name", name_node);
        assert!(node.captures.contains_key("name"));
    }

    #[test]
    fn test_tree_pretty() {
        let leaf = TreeNode::terminal("Ident", Span::new(3, 6));
        let parent = TreeNode::internal("FuncDef", Span::new(0, 8), vec![leaf]);
        let out = parent.pretty("fn foo()");
        assert!(out.contains("FuncDef"));
        assert!(out.contains("Ident"));
    }
}
