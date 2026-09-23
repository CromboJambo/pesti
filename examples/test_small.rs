use pesti_structural_tokenizer::StructuralTokenizer;

fn main() {
    let src = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
    let tok = StructuralTokenizer::new();
    let tokens = tok.tokenize(src).expect("tokenize failed");
    println!("Tokens: {}", tokens.len());
}
