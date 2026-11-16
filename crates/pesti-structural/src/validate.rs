//! Grammar validation: well-foundedness, reachability, determinism checks.
//!
//! These checks ensure the grammar is structurally sound before it's used
//! for parsing or embedding feature extraction.

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::expr::Expr;
use crate::grammar::{Nonterminal, StructuralGrammar, TerminalKind};

/// Validation errors found in a grammar.
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("undefined nonterminal: {nt} (referenced in rule {rule})")]
    UndefinedNonterminal { nt: String, rule: String },

    #[error("unreachable rule: {nt}")]
    UnreachableRule { nt: String },

    #[error("left recursion in rule: {nt}")]
    LeftRecursion { nt: String },

    #[error("empty loop in rule: {nt} (expression can match empty string)")]
    EmptyLoop { nt: String },

    #[error("ambiguous ordered choice in rule: {rule} — alternatives may both match the same input")]
    AmbiguousChoice { rule: String },

    #[error("start rule not defined: {rule}")]
    StartRuleNotDefined { rule: String },

    #[error("annotation references undefined rule: {rule}")]
    AnnotationUndefinedRule { rule: String },

    #[error("annotation references undefined capture: {capture} (in rule {rule})")]
    AnnotationUndefinedCapture { capture: String, rule: String },
}

/// Result of validating a grammar.
#[derive(Debug)]
pub struct ValidationResult {
    /// Whether the grammar is valid.
    pub valid: bool,
    /// Errors found (empty if valid).
    pub errors: Vec<ValidationError>,
    /// Warnings (non-fatal issues).
    pub warnings: Vec<String>,
}

impl ValidationResult {
    fn new() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    fn add_error(&mut self, err: ValidationError) {
        self.valid = false;
        self.errors.push(err);
    }

    fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
}

/// Validate a structural grammar.
///
/// Performs all 7 checks from the spec:
/// 1. All nonterminals referenced are defined
/// 2. All rules reachable from start rule
/// 3. No left recursion
/// 4. No empty loops (Repeat of ε-matching expression)
/// 5. Ordered choice determinism (conservative check)
/// 6. Start rule is defined
/// 7. Annotations reference valid rules and captures
pub fn validate(grammar: &StructuralGrammar) -> ValidationResult {
    let mut result = ValidationResult::new();

    // Check 6: Start rule is defined
    if !grammar.has_rule(&grammar.start) {
        result.add_error(ValidationError::StartRuleNotDefined {
            rule: grammar.start.0.clone(),
        });
        return result; // Can't proceed without a valid start rule
    }

    // Check 1: All referenced nonterminals are defined
    for (nt, expr) in &grammar.productions {
        let mut refs = Vec::new();
        expr.collect_nonterminals(&mut refs);
        let ref_owners: Vec<(String, String)> = refs
            .iter()
            .filter(|r| !grammar.has_rule(r))
            .map(|r| (r.0.clone(), nt.0.clone()))
            .collect();
        for (ref_nt, rule) in ref_owners {
            result.add_error(ValidationError::UndefinedNonterminal {
                nt: ref_nt,
                rule,
            });
        }
    }

    // Check 2: All rules reachable from start
    let reachable = collect_reachable(grammar, &grammar.start);
    for nt in grammar.nonterminals() {
        if !reachable.contains(nt.0.as_str()) {
            result.add_error(ValidationError::UnreachableRule {
                nt: nt.0.clone(),
            });
        }
    }

    // Check 3: No left recursion (direct or indirect)
    check_left_recursion(grammar, &mut result);

    // Check 4: No empty loops
    for (nt, expr) in &grammar.productions {
        check_empty_loops(expr, &nt.0, &mut result);
    }

    // Check 5: Ordered choice determinism (conservative)
    for (nt, expr) in &grammar.productions {
        check_choice_determinism(expr, &nt.0, grammar, &mut result);
    }

    // Check 7: Annotations reference valid rules and captures
    for (nt, annotation) in &grammar.annotations {
        if !grammar.has_rule(nt) {
            result.add_error(ValidationError::AnnotationUndefinedRule {
                rule: nt.0.clone(),
            });
            continue;
        }

        // Check that embed_fields reference existing captures
        if let Some(expr) = grammar.rule(nt) {
            let mut captures = Vec::new();
            expr.collect_captures(&mut captures);
            for field in &annotation.embed_fields {
                if !captures.contains(field) {
                    result.add_error(ValidationError::AnnotationUndefinedCapture {
                        capture: field.clone(),
                        rule: nt.0.clone(),
                    });
                }
            }
        }
    }

    result
}

/// Collect all nonterminal names reachable from the start rule.
fn collect_reachable(grammar: &StructuralGrammar, start: &Nonterminal) -> BTreeSet<String> {
    let mut reachable = BTreeSet::new();
    let mut stack = vec![start.0.clone()];

    while let Some(nt_name) = stack.pop() {
        if !reachable.insert(nt_name.clone()) {
            continue; // Already visited
        }
        let nt = Nonterminal(nt_name);
        if let Some(expr) = grammar.rule(&nt) {
            let mut refs = Vec::new();
            expr.collect_nonterminals(&mut refs);
            for ref_nt in refs {
                if grammar.has_rule(&ref_nt) && !reachable.contains(&ref_nt.0) {
                    stack.push(ref_nt.0.clone());
                }
            }
        }
    }

    reachable
}

/// Check for left recursion (direct or indirect) in PEG sense.
///
/// In PEGs, left recursion only matters if a nonterminal can appear as the
/// FIRST thing derived (before any terminal is consumed). Mutual recursion
/// like `Expr -> Call -> Expr` where `Call` starts with a terminal is fine.
fn check_left_recursion(grammar: &StructuralGrammar, result: &mut ValidationResult) {
    // Build a first-position call graph: nt -> set of nts that can appear
    // as the first nonterminal derived (before any terminal).
    let mut first_graph: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (nt, expr) in &grammar.productions {
        let mut refs = BTreeSet::new();
        collect_first_nonterminals(expr, &mut refs);
        // Only include edges to defined rules
        refs.retain(|r| grammar.has_rule(&Nonterminal(r.clone())));
        first_graph.insert(nt.0.clone(), refs);
    }

    // For each rule, check if it can reach itself through the first-position graph
    for (nt, _) in &grammar.productions {
        if reaches_self(&first_graph, &nt.0) {
            result.add_error(ValidationError::LeftRecursion {
                nt: nt.0.clone(),
            });
        }
    }
}

/// Collect nonterminals that can appear in first position of an expression
/// (before any terminal is consumed).
fn collect_first_nonterminals(expr: &Expr, out: &mut BTreeSet<String>) {
    match expr {
        Expr::Terminal(_) | Expr::CharRange(_, _) => {}
        Expr::Nonterminal(nt) => {
            out.insert(nt.0.clone());
        }
        Expr::Sequence(items) => {
            for item in items {
                let mut item_nts = BTreeSet::new();
                collect_first_nonterminals(item, &mut item_nts);
                out.extend(item_nts);
                // If this item cannot match empty, no need to look further
                if !item.can_match_empty() {
                    break;
                }
            }
        }
        Expr::OrderedChoice(alts) => {
            for alt in alts {
                let mut alt_nts = BTreeSet::new();
                collect_first_nonterminals(alt, &mut alt_nts);
                out.extend(alt_nts);
            }
        }
        Expr::Optional(inner) | Expr::Repeat(inner) | Expr::Capture(_, inner) => {
            collect_first_nonterminals(inner, out);
        }
    }
}

/// Check if `from` can reach itself through the call graph (DFS).
fn reaches_self(graph: &BTreeMap<String, BTreeSet<String>>, from: &str) -> bool {
    let mut visited = BTreeSet::new();
    let mut stack = vec![from.to_string()];

    while let Some(current) = stack.pop() {
        if !visited.insert(current.clone()) {
            continue;
        }
        if let Some(successors) = graph.get(&current) {
            for succ in successors {
                if succ == from {
                    return true; // Found a cycle back to `from`
                }
                if !visited.contains(succ) {
                    stack.push(succ.clone());
                }
            }
        }
    }

    false
}

/// Check for empty loops: Repeat(expr) where expr can match empty string.
fn check_empty_loops(expr: &Expr, rule_name: &str, result: &mut ValidationResult) {
    match expr {
        Expr::Repeat(inner) => {
            if inner.can_match_empty() {
                result.add_error(ValidationError::EmptyLoop {
                    nt: rule_name.to_string(),
                });
            }
            check_empty_loops(inner, rule_name, result);
        }
        Expr::Sequence(items) | Expr::OrderedChoice(items) => {
            for item in items {
                check_empty_loops(item, rule_name, result);
            }
        }
        Expr::Optional(inner) | Expr::Capture(_, inner) => {
            check_empty_loops(inner, rule_name, result);
        }
        Expr::Terminal(_) | Expr::Nonterminal(_) | Expr::CharRange(_, _) => {}
    }
}

/// Conservative determinism check for ordered choices.
fn check_choice_determinism(
    expr: &Expr,
    rule_name: &str,
    grammar: &StructuralGrammar,
    result: &mut ValidationResult,
) {
    match expr {
        Expr::OrderedChoice(alts) => {
            // Get first terminals for each alternative
            let mut first_sets: Vec<Vec<TerminalKind>> = Vec::new();
            for alt in alts {
                let mut firsts = Vec::new();
                resolve_first_terminals(alt, grammar, &mut firsts);
                first_sets.push(firsts);
            }

            // Check for overlap between alternatives
            for i in 0..first_sets.len().saturating_sub(1) {
                for j in (i + 1)..first_sets.len() {
                    for t1 in &first_sets[i] {
                        for t2 in &first_sets[j] {
                            if t1 == t2 {
                                result.add_warning(format!(
                                    "potential ambiguity in rule '{rule_name}': alternatives may both match terminal {t1}"
                                ));
                                return; // One warning per rule is enough
                            }
                        }
                    }
                }
            }

            // Recurse into alternatives
            for alt in alts {
                check_choice_determinism(alt, rule_name, grammar, result);
            }
        }
        Expr::Sequence(items) => {
            for item in items {
                check_choice_determinism(item, rule_name, grammar, result);
            }
        }
        Expr::Optional(inner) | Expr::Repeat(inner) | Expr::Capture(_, inner) => {
            check_choice_determinism(inner, rule_name, grammar, result);
        }
        Expr::Terminal(_) | Expr::Nonterminal(_) | Expr::CharRange(_, _) => {}
    }
}

/// Resolve first terminals for an expression, following nonterminal references (one level).
fn resolve_first_terminals(
    expr: &Expr,
    grammar: &StructuralGrammar,
    out: &mut Vec<TerminalKind>,
) {
    match expr {
        Expr::Terminal(tk) => out.push(tk.clone()),
        Expr::CharRange(_, _) => {}
        Expr::Sequence(items) => {
            if let Some(first) = items.first() {
                resolve_first_terminals(first, grammar, out);
            }
        }
        Expr::OrderedChoice(alts) => {
            for alt in alts {
                resolve_first_terminals(alt, grammar, out);
            }
        }
        Expr::Optional(inner) | Expr::Repeat(inner) | Expr::Capture(_, inner) => {
            resolve_first_terminals(inner, grammar, out);
        }
        Expr::Nonterminal(nt) => {
            if let Some(rule_expr) = grammar.rule(nt) {
                // Avoid infinite recursion: only follow one level
                match rule_expr {
                    Expr::Terminal(tk) => out.push(tk.clone()),
                    Expr::Sequence(items) => {
                        if let Some(first) = items.first() {
                            if let Expr::Terminal(tk) = first {
                                out.push(tk.clone());
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_peg;

    #[test]
    fn test_valid_grammar() {
        let src = r#"grammar test;
Start = { 'fn' name: Ident }
"#;
        let g = parse_peg(src).unwrap();
        let result = validate(&g);
        assert!(result.valid, "errors: {:?}", result.errors);
    }

    #[test]
    fn test_undefined_nonterminal() {
        let src = r#"grammar test;
Start = { UndefinedRule }
"#;
        let g = parse_peg(src).unwrap();
        let result = validate(&g);
        assert!(!result.valid);
        assert!(matches!(
            result.errors[0],
            ValidationError::UndefinedNonterminal { .. }
        ));
    }

    #[test]
    fn test_unreachable_rule() {
        let src = r#"grammar test;
Start = { 'a' }
Unused = { 'b' }
"#;
        let g = parse_peg(src).unwrap();
        let result = validate(&g);
        assert!(!result.valid);
        assert!(matches!(
            result.errors[0],
            ValidationError::UnreachableRule { .. }
        ));
    }

    #[test]
    fn test_left_recursion() {
        // Direct left recursion: A -> A ...
        let src = r#"grammar test;
Start = { 'a' A }
A = { A 'b' }
"#;
        let g = parse_peg(src).unwrap();
        let result = validate(&g);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| matches!(e, ValidationError::LeftRecursion { .. })));
    }

    #[test]
    fn test_empty_loop() {
        // Repeat of something that can match empty
        let src = r#"grammar test;
Start = { ('a'?)* }
"#;
        let g = parse_peg(src).unwrap();
        let result = validate(&g);
        assert!(!result.valid);
        assert!(result.errors.iter().any(|e| matches!(e, ValidationError::EmptyLoop { .. })));
    }

    #[test]
    fn test_annotation_undefined_rule() {
        let src = r#"grammar test;
Start = { 'a' }

annotations {
    NonExistent { role: "definition" }
}
"#;
        let g = parse_peg(src).unwrap();
        let result = validate(&g);
        assert!(!result.valid);
        assert!(matches!(
            result.errors[0],
            ValidationError::AnnotationUndefinedRule { .. }
        ));
    }

    #[test]
    fn test_annotation_undefined_capture() {
        let src = r#"grammar test;
Start = { 'fn' name: Ident }

annotations {
    Start { role: "definition", embed_fields: ["name", "missing"] }
}
"#;
        let g = parse_peg(src).unwrap();
        let result = validate(&g);
        assert!(!result.valid);
        assert!(matches!(
            result.errors[0],
            ValidationError::AnnotationUndefinedCapture { .. }
        ));
    }

    #[test]
    fn test_start_rule_not_defined() {
        let src = r#"grammar test;
A = { 'a' }
"#;
        // Manually set start to something undefined
        let mut g = parse_peg(src).unwrap();
        g.start = Nonterminal("NonExistent".into());
        let result = validate(&g);
        assert!(!result.valid);
        assert!(matches!(
            result.errors[0],
            ValidationError::StartRuleNotDefined { .. }
        ));
    }

    #[test]
    fn test_complex_valid_grammar() {
        let src = r#"grammar rust_structural;
Module       = { item: Item* }
Item         = { kind: FuncDef / StructDef }
FuncDef      = { Visibility?, 'fn', name: Ident, params: ParamList, body: Block }
StructDef    = { 'struct', name: Ident, fields: '{' Field* '}' }
ParamList    = { '(' param: Param* ')' }
Param        = { name: Ident, ':', Type }
Type         = { Ident / Path }
Path         = { seg: Ident ('::' seg: Ident)* }
Block        = { '{' stmt: Stmt* '}' }
Stmt         = { Expr }
Expr         = { Call / Literal }
Call         = { name: Ident, '(' arg: Expr* ')' }
Literal      = { IntLit / FloatLit / StrLit }
Visibility   = { 'pub' }
Field        = { name: Ident, ':', Type }

annotations {
    FuncDef   { role: "definition", embed_fields: ["name", "params"], class: "declaration" }
    StructDef { role: "definition", embed_fields: ["name", "fields"], class: "declaration" }
    Call      { role: "expression", embed_fields: ["name", "arg"], class: "invocation" }
}
"#;
        let g = parse_peg(src).unwrap();
        let result = validate(&g);
        // This grammar should be valid (warnings about ambiguity are OK)
        assert!(result.valid, "errors: {:?}, warnings: {:?}", result.errors, result.warnings);
    }
}
