//! Debug: show exactly where parsing fails on a given file.
use std::env;
use std::fs;
use pesti_structural::evaluator::PegEvaluator;
use pesti_structural::lexer::tokenize_rust;
use pesti_structural::parser::parse_peg;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: debug_parse <file.rs>");
        std::process::exit(1);
    }

    let path = &args[1];
    let source = fs::read_to_string(path).expect("read file");
    let grammar_src = include_str!("../grammars/rust.peg");
    let grammar = parse_peg(grammar_src).expect("parse grammar");
    let (tokens, lex_errors) = tokenize_rust(&source);

    if !lex_errors.is_empty() {
        println!("Lexer errors: {:?}", lex_errors);
    }

    let eval = PegEvaluator::new(&grammar, &tokens);
    let result = eval.parse();

    println!("File: {}", path);
    println!("Tokens: {}, Errors in parse: {}", tokens.len(), result.errors.len());
    for err in &result.errors {
        println!("  ERROR at [{}..{}]: {} (expected: {:?})",
            err.span.start, err.span.end, err.message, err.expected);
        // Show context around the error
        let line_start = source[..err.span.start.min(source.len())]
            .rfind('\n')
            .map(|p| p + 1)
            .unwrap_or(0);
        let line_end = source[line_start..]
            .find('\n')
            .map(|p| line_start + p)
            .unwrap_or(source.len());
        println!("  CONTEXT: {:?}", &source[line_start..line_end.min(line_start + 120)]);
        // Show tokens around error position
        let err_token_idx = tokens.iter().position(|t| t.span.start >= err.span.start).unwrap_or(tokens.len() - 1);
        for i in (err_token_idx.saturating_sub(3)..=err_token_idx.min(tokens.len().saturating_sub(1))).collect::<Vec<_>>() {
            let t = &tokens[i];
            println!("    tok[{}]: {:?} '{}' [{}..{}]", i, t.kind, t.text, t.span.start, t.span.end);
        }
    }

    if result.tree.is_some() {
        println!("Parse: SUCCESS");
    } else {
        println!("Parse: FAILED (no tree)");
    }
}
