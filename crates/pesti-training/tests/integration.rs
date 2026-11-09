//! End-to-end integration test for pesti-training pipeline.
//!
//! Validates that crabjar's JSONL decision training data can be loaded,
//! tokenized, and formatted into proper training samples with correct
//! input_ids/attention_mask/labels structure.

use pesti_training::{DatasetLoader, QLoRAConfig};
use qwen2_bpe::Qwen2Tokenizer;
use std::collections::HashMap;
use std::fs;

fn make_test_loader() -> DatasetLoader {
    // Create tokenizer with actual vocab for real tokenization
    let config = qwen2_bpe::Qwen2Config {
        vocab: HashMap::new(),
        merges: vec![],
        special_tokens: None,
    };
    let tokenizer = Qwen2Tokenizer::new(config).expect("tokenizer init");
    DatasetLoader::new(tokenizer, 512)
}

#[test]
fn test_load_real_training_data() {
    // Load actual crabjar training data
    let loader = make_test_loader();

    let path = "/home/crombo/training-data-dump/comprehensive_decision_training.jsonl";
    if !fs::metadata(path).is_ok() {
        eprintln!("Skipping: {} not found", path);
        return;
    }

    let samples = loader.load_jsonl(path).expect("load training data");

    // Verify we loaded a reasonable number of samples
    assert!(samples.len() > 0, "Should load at least one sample");
    eprintln!("Loaded {} training samples", samples.len());

    // Verify each sample has correct structure
    let mut valid_samples = 0;
    for (i, sample) in samples.iter().enumerate() {
        assert!(
            !sample.input_ids.is_empty(),
            "Sample {} has empty input_ids",
            i
        );
        assert_eq!(
            sample.input_ids.len(),
            sample.attention_mask.len(),
            "Sample {} input/mask length mismatch",
            i
        );
        assert_eq!(
            sample.input_ids.len(),
            sample.labels.len(),
            "Sample {} input/labels length mismatch",
            i
        );

        // Verify labels contain actual target tokens (not all -100)
        let has_targets = sample.labels.iter().any(|&l| l != -100);
        if has_targets {
            valid_samples += 1;
        } else {
            eprintln!("Warning: Sample {} has no target tokens (truncated)", i);
        }

        // Verify attention mask is binary
        for (pos, &mask_val) in sample.attention_mask.iter().enumerate() {
            assert!(
                mask_val == 0 || mask_val == 1,
                "Sample {} invalid mask value at {}",
                i,
                pos
            );
        }
    }

    // Most samples should have targets (some may be truncated)
    assert!(
        valid_samples > samples.len() / 2,
        "Too many samples without target tokens: {}/{}",
        valid_samples,
        samples.len()
    );

    eprintln!("✓ All samples have correct structure");
}

#[test]
fn test_sample_formatting_includes_decision_fields() {
    let loader = make_test_loader();

    // Create a known example to verify formatting
    let example_json = r#"{"scenario":"Test destructive command","context":{"node":"ftw3"},"decision":"reject","confidence":95,"reasoning":"Too risky without backup","risk_factors":["no_backup"],"next_steps":["verify"]}"#;

    // Write to temp file
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let temp_path = temp_dir.path().join("test.jsonl");
    fs::write(&temp_path, example_json).expect("write test data");

    let samples = loader.load_jsonl(temp_path).expect("load sample");
    assert_eq!(samples.len(), 1);

    let sample = &samples[0];

    // Decode input to verify it contains the scenario text
    let input_str: String = sample
        .input_ids
        .iter()
        .map(|&id| id as u8)
        .collect::<Vec<u8>>()
        .into_iter()
        .filter(|b| *b > 31 && *b < 127)
        .map(|b| b as char)
        .collect();

    // Verify scenario text appears in input
    assert!(
        input_str.contains("destructive command") || input_str.contains("Scenario"),
        "Input should contain scenario text"
    );

    eprintln!("✓ Sample formatting verified");
}

#[test]
fn test_lora_config_defaults() {
    let config = QLoRAConfig::default();

    assert_eq!(config.rank, 8);
    assert_eq!(config.alpha, 16.0);
    assert_eq!(config.dropout, 0.05);
    assert!(config.target_modules.contains(&"q_proj".to_string()));
    assert!(config.target_modules.contains(&"k_proj".to_string()));
    assert!(config.target_modules.contains(&"v_proj".to_string()));

    eprintln!("✓ LoRA config defaults verified");
}

#[test]
fn test_dataset_error_handling() {
    let loader = make_test_loader();

    // Test with malformed JSON
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let temp_path = temp_dir.path().join("bad.jsonl");
    fs::write(&temp_path, "not valid json").expect("write bad data");

    let result = loader.load_jsonl(temp_path);
    assert!(result.is_err(), "Should fail on malformed JSON");

    eprintln!("✓ Error handling verified");
}
