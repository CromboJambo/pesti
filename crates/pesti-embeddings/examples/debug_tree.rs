use pesti_structural::evaluator::PegEvaluator;
use pesti_structural::lexer::tokenize_for_language;
use pesti_structural::parser::parse_peg_file;
use std::collections::BTreeSet;

fn collect_labels(node: &pesti_structural::tree::TreeNode, labels: &mut BTreeSet<String>) {
    labels.insert(node.label.clone());
    for child in &node.children {
        collect_labels(child, labels);
    }
}

fn main() {
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let grammar_path = workspace_root.join("crates").join("pesti-structural").join("grammars").join("rust.peg");
    
    let grammar = parse_peg_file(&grammar_path).unwrap();
    
    let source = r#"pub fn add(a: i32, b: i32) -> i32 {
    a + b
}"#;
    
    let (tokens, _) = tokenize_for_language(source, "rust");
    println!("Tokens: {}", tokens.len());
    for t in &tokens {
        println!("  {:?} '{}'", t.kind, t.text);
    }
    
    let evaluator = PegEvaluator::new(&grammar, &tokens);
    let result = evaluator.parse_with_recovery();
    
    match result.tree {
        Some(tree) => {
            println!("\nTree root: {}", tree.label);
            let mut labels = BTreeSet::new();
            collect_labels(&tree, &mut labels);
            println!("\nAll tree labels:");
            for l in &labels {
                println!("  {}", l);
            }
            println!("\nTree structure:");
            println!("{}", tree.pretty(source));
        }
        None => println!("No tree: {:?}", result.errors),
    }
}
