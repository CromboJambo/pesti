//! Comprehensive tests for the TRL (Transformer Reinforcement Learning) module.

use pesti_runner::peft::{Adapter, AdapterConfig, LoRAAdapter};
use pesti_runner::trl::{
    Batch, Callbacks, CrossEntropyLoss, Dataset, InMemoryDataset, KLDivergenceLoss,
    LoggingCallback, LossFunction, Metrics, OptimizerConfig, PairwiseRankingLoss, State,
    TrainingConfig,
};

// Helper to create a dataset with labels - just construct directly
fn make_dataset(input_ids: Vec<Vec<f32>>, labels: Option<Vec<Vec<u32>>>) -> Box<dyn Dataset> {
    Box::new(InMemoryDataset::new(input_ids, labels))
}

#[test]
fn test_batch_creation() {
    let input_ids = vec![1.0, 2.0, 3.0, 4.0];
    let attention_mask = vec![1u8, 1u8, 1u8, 1u8];

    let batch = Batch::new(input_ids.clone(), attention_mask.clone());

    assert_eq!(batch.input_ids, input_ids);
    assert_eq!(batch.attention_mask, attention_mask);
    assert!(batch.labels.is_none());
}

#[test]
fn test_batch_with_labels() {
    let input_ids = vec![1.0, 2.0, 3.0];
    let labels = vec![1u32, 2u32, 3u32];

    let batch = Batch::new(input_ids, vec![1u8, 1u8, 1u8]).with_labels(labels.clone());

    assert_eq!(batch.labels, Some(labels));
}

#[test]
fn test_batch_size_calculation() {
    // Single sequence of length 10
    let input_ids = vec![1.0; 10];
    let attention_mask = vec![1u8; 10];

    let batch = Batch::new(input_ids, attention_mask);
    assert_eq!(batch.batch_size(), 1);
    assert_eq!(batch.seq_len(), 10);
}

#[test]
fn test_in_memory_dataset() {
    let input_ids = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];

    let labels = vec![vec![1u32, 2u32, 3u32], vec![4u32, 5u32, 6u32]];

    let dataset = make_dataset(input_ids.clone(), Some(labels));

    assert_eq!(dataset.len(), input_ids.len());

    // Test getting samples
    let sample0 = dataset.get(0).unwrap();
    assert_eq!(sample0.input_ids, vec![1.0, 2.0, 3.0]);
    assert_eq!(sample0.labels, Some(vec![1u32, 2u32, 3u32]));

    let sample2 = dataset.get(2).unwrap();
    assert_eq!(sample2.input_ids, vec![7.0, 8.0, 9.0]);
    // Sample 2 has no labels (only 2 label entries provided)
    assert!(sample2.labels.is_none());
}

#[test]
fn test_in_memory_dataset_without_labels() {
    let input_ids = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    let dataset = make_dataset(input_ids.clone(), None);

    assert_eq!(dataset.len(), 2);

    let sample = dataset.get(0).unwrap();
    assert!(sample.labels.is_none());
}

#[test]
fn test_cross_entropy_loss() {
    let loss_fn = CrossEntropyLoss::default();

    // Simple case: batch_size=2, vocab_size=3
    // logits: [logit_0_0, logit_0_1, logit_0_2, logit_1_0, logit_1_1, logit_1_2]
    let logits = vec![
        2.0, 1.0, 0.0, // Sample 0
        1.0, 2.0, 0.0, // Sample 1
    ];

    // targets: token IDs [0, 1]
    let targets = vec![0u32, 1u32];

    let loss = loss_fn.compute(&logits, &targets);

    // Loss should be positive and finite
    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_cross_entropy_loss_one_hot() {
    let loss_fn = CrossEntropyLoss::default();

    // When model is perfect (high logit for correct class)
    let logits = vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0];
    let targets = vec![0u32, 1u32];

    let loss = loss_fn.compute(&logits, &targets);

    // Loss should be very low (close to 0)
    assert!(loss < 0.1);
}

#[test]
fn test_cross_entropy_loss_high() {
    let loss_fn = CrossEntropyLoss::default();

    // When model is wrong (low logit for correct class)
    let logits = vec![0.0, 0.0, 10.0, 10.0, 0.0, 0.0];
    let targets = vec![0u32, 1u32];

    let loss = loss_fn.compute(&logits, &targets);

    // Loss should be high
    assert!(loss > 2.0);
}

#[test]
fn test_kl_divergence_loss() {
    let loss_fn = KLDivergenceLoss::default();

    // KL divergence with one-hot targets reduces to cross-entropy
    let logits = vec![2.0, 1.0, 0.0, 1.0, 2.0, 0.0];
    let targets = vec![0u32, 1u32];

    let loss = loss_fn.compute(&logits, &targets);

    // KL divergence should be non-negative (can be 0 for perfect match)
    eprintln!("KL divergence loss: {}", loss);
    assert!(loss >= -1e-6, "KL divergence is negative: {}", loss); // Allow small numerical error
    assert!(loss.is_finite());
}

#[test]
fn test_pairwise_ranking_loss() {
    let loss_fn = PairwiseRankingLoss::default();

    // Simulated pairwise preferences: target=1 means chosen > rejected
    let logits = vec![
        1.0, 2.0, // Sample 0: logit_chosen=2.0, logit_rejected=1.0
        2.0, 1.0, // Sample 1: logit_chosen=1.0, logit_rejected=2.0 (reversed)
    ];
    let targets = vec![1u32, 0u32]; // Preferences

    let loss = loss_fn.compute(&logits, &targets);

    assert!(loss >= 0.0);
}

#[test]
fn test_metrics_creation() {
    let metrics = Metrics::new(0.5).with_perplexity(1.6487);

    assert_eq!(metrics.loss, 0.5);
    assert_eq!(metrics.perplexity, Some(1.6487));
    assert!(metrics.accuracy.is_none());
}

#[test]
fn test_metrics_with_accuracy() {
    let metrics = Metrics::new(0.3).with_accuracy(0.95);

    assert_eq!(metrics.loss, 0.3);
    assert_eq!(metrics.accuracy, Some(0.95));
}

#[test]
fn test_metrics_custom() {
    let metrics = Metrics::new(0.5)
        .with_custom("bleu", 0.42)
        .with_custom("rouge", 0.38);

    assert_eq!(metrics.custom.get("bleu"), Some(&0.42));
    assert_eq!(metrics.custom.get("rouge"), Some(&0.38));
}

#[test]
fn test_state_new() {
    let state = State::new();

    assert_eq!(state.epoch, 0);
    assert_eq!(state.global_step, 0);
    assert_eq!(state.epoch_step, 0);
    assert!(state.losses.is_empty());
    assert!(state.eval_metrics.is_empty());
    assert!(!state.is_finished);
}

#[test]
fn test_state_record_loss() {
    let mut state = State::new();

    state.record_loss(1.0);
    state.record_loss(0.8);
    state.record_loss(0.6);

    assert_eq!(state.losses.len(), 3);
    assert_eq!(state.losses[0], 1.0);
    assert_eq!(state.losses[1], 0.8);
    assert_eq!(state.losses[2], 0.6);
}

#[test]
fn test_state_avg_loss() {
    let mut state = State::new();

    state.record_loss(1.0);
    state.record_loss(2.0);
    state.record_loss(3.0);

    assert_eq!(state.avg_loss(), Some(2.0));
}

#[test]
fn test_state_increment_epoch() {
    let mut state = State::new();

    state.increment_epoch();
    assert_eq!(state.epoch, 1);

    state.increment_epoch();
    state.increment_epoch();
    assert_eq!(state.epoch, 3);
}

#[test]
fn test_state_increment_step() {
    let mut state = State::new();

    state.increment_step(5);
    assert_eq!(state.global_step, 5);
    assert_eq!(state.epoch_step, 5);

    state.increment_step(10);
    assert_eq!(state.global_step, 15);
    assert_eq!(state.epoch_step, 15);
}

#[test]
fn test_state_record_eval() {
    let mut state = State::new();

    let metrics1 = Metrics::new(0.5).with_perplexity(1.65);
    let metrics2 = Metrics::new(0.4).with_perplexity(1.49);

    state.record_eval(metrics1.clone());
    state.record_eval(metrics2.clone());

    assert_eq!(state.eval_metrics.len(), 2);
    assert_eq!(state.eval_metrics[0].loss, 0.5);
    assert_eq!(state.eval_metrics[1].loss, 0.4);
}

#[test]
fn test_state_best_eval() {
    let mut state = State::new();

    // Record metrics with different losses (lower is better)
    state.record_eval(Metrics::new(0.5).with_perplexity(1.65));
    state.record_eval(Metrics::new(0.3).with_perplexity(1.35));
    state.record_eval(Metrics::new(0.7).with_perplexity(2.01));

    let best = state.best_eval().unwrap();
    assert_eq!(best.loss, 0.3); // Best (lowest) loss
}

#[test]
fn test_state_finish() {
    let mut state = State::new();

    assert!(!state.is_finished);
    state.finish();
    assert!(state.is_finished);
}

#[test]
fn test_callbacks_empty() {
    let mut callbacks = Callbacks::new();
    let state = State::new();

    // Empty callbacks should not panic
    callbacks.on_train_start(&state);
    callbacks.on_epoch_end(&state);
    callbacks.on_step_end(&state, 1.0);
    callbacks.on_eval_start(&state);
    callbacks.on_eval_end(&state, &Metrics::new(0.5));
}

#[test]
fn test_callbacks_with_logging() {
    let mut callbacks = Callbacks::new();
    callbacks.add(LoggingCallback::new("test.log"));

    let mut state = State::new();
    state.record_loss(1.0);
    state.record_loss(0.8);
    state.increment_epoch();

    // Should not panic
    callbacks.on_epoch_end(&state);
}

#[test]
fn test_training_config_default() {
    let config = TrainingConfig::default();

    assert_eq!(config.num_epochs, 3);
    assert!((config.learning_rate - 2e-4).abs() < 1e-6);
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.gradient_accumulation_steps, 1);
}

#[test]
fn test_training_config_builder() {
    let config = TrainingConfig::default()
        .with_num_epochs(5)
        .with_learning_rate(1e-4)
        .with_batch_size(8)
        .with_gradient_accumulation(4)
        .with_fp16(true);

    assert_eq!(config.num_epochs, 5);
    assert!((config.learning_rate - 1e-4).abs() < 1e-6);
    assert_eq!(config.batch_size, 8);
    assert_eq!(config.gradient_accumulation_steps, 4);
    assert!(config.use_fp16);
}

#[test]
fn test_optimizer_config_default() {
    let config = OptimizerConfig::default();

    assert_eq!(config.learning_rate, 2e-4);
    assert_eq!(config.weight_decay, 0.1);
    assert!((config.beta1 - 0.9).abs() < 1e-6);
    assert!((config.beta2 - 0.95).abs() < 1e-6);
}

#[test]
fn test_lora_adapter_forward_shape() {
    let config = AdapterConfig::lora(8, 16.0);
    let adapter = LoRAAdapter::new_random(64, 128, &config).unwrap();

    // Input: batch_size=2, features=64
    let x: Vec<f32> = (0..2 * 64).map(|i| i as f32).collect();

    let output = adapter.forward(&x, 2).unwrap();

    // Output should be [batch_size, out_features] = [2, 128]
    assert_eq!(output.len(), 2 * 128);
}

#[test]
fn test_lora_adapter_properties() {
    let config = AdapterConfig::lora(16, 32.0);
    let adapter = LoRAAdapter::new_random(256, 512, &config).unwrap();

    assert_eq!(adapter.rank(), 16);
    assert_eq!(adapter.scaling(), 32.0);
    assert_eq!(adapter.in_features, 256);
    assert_eq!(adapter.out_features, 512);
}

#[test]
fn test_dataset_iterator() {
    let input_ids = vec![vec![1.0, 2.0], vec![3.0, 4.0]];

    // Use concrete type for iterator (requires Sized)
    let dataset: InMemoryDataset = InMemoryDataset {
        input_ids: input_ids.clone(),
        labels: None,
    };

    let mut count = 0;
    for _batch in dataset.iter() {
        count += 1;
    }

    assert_eq!(count, 2);
}
