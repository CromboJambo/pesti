//! Validate building the real Qwen2.5 BPE tokenizer from GGUF metadata.
//!
//! Encodes a known prompt and compares against llama.cpp reference token IDs:
//!   "Once upon a time in the land of Rust," ->
//!   [12522, 5193, 264, 882, 304, 279, 4268, 315, 33789, 11]
use pesti_gguf::parser::parse_gguf;
use pesti_gguf::types::GgufKvValue;
use std::collections::HashMap;
use std::path::Path;
use tokenizers::decoders::Decoder;
use tokenizers::models::Bpe;
use tokenizers::models::BpeBuilder;
use tokenizers::pre_tokenizers::PreTokenizer;
use tokenizers::Tokenizer;

const MODEL: &str =
    "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";

/// Qwen2 byte-level BPE pre-tokenizer regex (from HuggingFace Qwen2 tokenizer.json).
const QWEN2_PRE_TOKENIZER: &str = r"(?i:'s'|'t'|'ve'|'re'|'ll'|'d)|\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+?| ?";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let header = parse_gguf(Path::new(MODEL))?;

    // 1. tokens: array of strings (byte-level, may contain non-UTF-8)
    let tokens = match header.get_kv::<GgufKvValue>("tokenizer.ggml.tokens") {
        Some(GgufKvValue::Array(arr)) => arr,
        _ => return Err("missing tokenizer.ggml.tokens".into()),
    };
    let n_tokens = tokens.len();

    // 2. merges: array of "a b" strings
    let merges = match header.get_kv::<GgufKvValue>("tokenizer.ggml.merges") {
        Some(GgufKvValue::Array(arr)) => arr,
        _ => return Err("missing tokenizer.ggml.merges".into()),
    };

    // Build vocab (token -> id) and merges ((left, right) pairs)
    let mut vocab: HashMap<String, u32> = HashMap::with_capacity(n_tokens);
    for (i, t) in tokens.iter().enumerate() {
        if let GgufKvValue::String(s) = t {
            vocab.insert(s.clone(), i as u32);
        }
    }
    let mut merge_pairs: Vec<(String, String)> = Vec::with_capacity(merges.len());
    for m in merges.iter() {
        if let GgufKvValue::String(s) = m {
            if let Some((a, b)) = s.split_once(' ') {
                merge_pairs.push((a.to_string(), b.to_string()));
            }
        }
    }

    println!("[build] vocab={} merges={}", vocab.len(), merge_pairs.len());

    // 3. Build the BPE model
    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merge_pairs)
        .build()
        .map_err(|e| format!("BpeBuilder failed: {e}"))?;

    // 4. Assemble tokenizer: BPE + ByteLevel decoder + Qwen2 pre-tokenizer
    let mut tokenizer = Tokenizer::new(tokenizers::models::ModelWrapper::BPE(bpe));
    tokenizer
        .with_decoder(Decoder::ByteLevel(true, true, true))
        .map_err(|e| format!("decoder: {e}"))?;
    tokenizer
        .with_pre_tokenizer(PreTokenizer::Split {
            pattern: PreTokenizer::SplitPattern::Regex(QWEN2_PRE_TOKENIZER.to_string()),
            behavior: tokenizers::pre_tokenizers::SplitBehavior::Removed,
            invert: false,
        })
        .map_err(|e| format!("pre_tokenizer: {e}"))?;

    // 5. Validate against llama.cpp reference
    let prompt = "Once upon a time in the land of Rust,";
    let expected = [12522, 5193, 264, 882, 304, 279, 4268, 315, 33789, 11];
    let enc = tokenizer.encode(prompt, true).map_err(|e| format!("encode: {e}"))?;
    let ids: Vec<u32> = enc.get_ids().to_vec();
    println!("[encode] {prompt:?}");
    println!("  PESTI:    {ids:?}");
    println!("  expected: {expected:?}");
    if ids == expected.to_vec() {
        println!("\u{2705} MATCH — tokenizer is correct");
    } else {
        println!("\u{274c} MISMATCH — investigate");
    }

    // 6. Round-trip decode
    let decoded = tokenizer.decode(&ids, true).map_err(|e| format!("decode: {e}"))?;
    println!("[decode] {decoded:?}");

    // 7. Check the week16 prompt too
    let prompt2 = "The quick brown fox jumps over the lazy dog.";
    let enc2 = tokenizer.encode(prompt2, true).map_err(|e| format!("encode: {e}"))?;
    println!("[encode2] {prompt2:?} -> {:?}", enc2.get_ids());

    Ok(())
}
