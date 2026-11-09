//! Tests for the PEFT (Parameter-Efficient Fine-Tuning) adapters.

use pesti_runner::peft::{Adapter, AdapterConfig, LoRAAdapter, LoRABuilder};

#[test]
fn test_lora_adapter_creation() {
    let config = AdapterConfig::lora(8, 16.0);
    let adapter = LoRAAdapter::new_random(512, 1024, &config).unwrap();

    assert_eq!(adapter.rank(), 8);
    assert_eq!(adapter.scaling(), 16.0);
    assert_eq!(adapter.in_features, 512);
    assert_eq!(adapter.out_features, 1024);
    assert!(adapter.is_initialized());
}

#[test]
fn test_lora_adapter_zeros() {
    let config = AdapterConfig::lora(8, 16.0);
    let adapter = LoRAAdapter::new_zeros(512, 1024, &config).unwrap();

    assert_eq!(adapter.rank(), 8);
    assert_eq!(adapter.scaling(), 16.0);
    // new_zeros initializes but doesn't set is_initialized to true
    // (that's done in init_random)
}

#[test]
fn test_lora_forward_pass() {
    let config = AdapterConfig::lora(8, 16.0);
    let adapter = LoRAAdapter::new_random(64, 128, &config).unwrap();

    // Create a simple input: batch_size=2, in_features=64
    let batch_size = 2;
    let x: Vec<f32> = (0..batch_size * 64).map(|i| i as f32).collect();

    let output = adapter.forward(&x, batch_size).unwrap();

    // Output should be [batch_size, out_features] = [2, 128]
    assert_eq!(output.len(), batch_size * 128);
}

#[test]
fn test_lora_builder() {
    let adapter = LoRABuilder::new()
        .with_rank(16)
        .with_scaling(32.0)
        .build_random(256, 512)
        .unwrap();

    assert_eq!(adapter.rank(), 16);
    assert_eq!(adapter.scaling(), 32.0);
}

#[test]
fn test_lora_merge() {
    let config = AdapterConfig::lora(8, 16.0);
    let adapter = LoRAAdapter::new_random(32, 64, &config).unwrap();

    // Create base weights: [out_features, in_features] = [64, 32]
    let base_weights: Vec<f32> = (0..64 * 32).map(|i| i as f32 * 0.1).collect();

    let (merged_weights, merged_bias) = adapter.merge_into(&base_weights, None).unwrap();

    // Merged weights should have the same shape as base weights
    assert_eq!(merged_weights.len(), 64 * 32);
    assert!(merged_bias.is_none());
}
