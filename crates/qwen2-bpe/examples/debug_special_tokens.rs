//! Debug special token decoding

use qwen2_bpe::{Qwen2Config, Qwen2Tokenizer, SpecialTokens};

fn main() {
    let config = Qwen2Config {
        vocab: vec![
            (72, vec![b'H']),
            (101, vec![b'e']),
            (108, vec![b'l']),
            (111, vec![b'o']),
            (32, vec![b' ']),
        ]
        .into_iter()
        .collect(),
        merges: vec![],
        special_tokens: Some(SpecialTokens {
            bos_token_id: 151643,
            eos_token_id: 151643,
            pad_token_id: Some(0),
        }),
    };

    let tokenizer = Qwen2Tokenizer::new(config).unwrap();
    
    // Decode with special tokens
    let decoded = tokenizer.decode(&[151643, 72, 101, 108, 108, 111, 151643]).unwrap();
    
    println!("Decoded: '{}'", decoded);
    println!("Contains BOS tag: {}", decoded.contains("<|begin_of_text|>"));
    println!("Contains EOS tag: {}", decoded.contains("<|end_of_text|>"));
}
