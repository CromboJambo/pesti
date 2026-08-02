//! Integration tests for GPU dispatch system.
//!
//! These tests verify the correctness of the dispatch layer against CPU baselines
//! using real model files from the conformance corpus.

use pesti_gguf::{GgufKvPair, GgufKvValue, GgufValueType};
use pesti_runner::kernel::dispatch::{DispatchContext, LinearDispatch};
use pesti_runner::gguf_weight_loader::load_gguf_weights;
use pesti_runner::model::CpuModel;
use half::f16;
use std::path::PathBuf;
use tempfile::tempdir;

fn kv_pair_str(key: &str, value: &str) -> GgufKvPair {
    GgufKvPair {
        key: key.to_string(),
        value_type: GgufValueType::String,
        value: GgufKvValue::String(value.to_string()),
    }
}

fn kv_pair_u64(key: &str, value: u32) -> GgufKvPair {
    GgufKvPair {
        key: key.to_string(),
        value_type: GgufValueType::Uint32,
        value: GgufKvValue::Uint32(value),
    }
}

fn kv_pair_f32(key: &str, value: f32) -> GgufKvPair {
    GgufKvPair {
        key: key.to_string(),
        value_type: GgufValueType::Float32,
        value: GgufKvValue::Float32(value),
    }
}

fn kv_pair_array(key: &str, items: Vec<GgufKvValue>) -> GgufKvPair {
    GgufKvPair {
        key: key.to_string(),
        value_type: GgufValueType::Array,
        value: GgufKvValue::Array(items),
    }
}

fn write_kv_value(buf: &mut Vec<u8>, value: &GgufKvValue) {
    match value {
        GgufKvValue::Uint8(v) => buf.push(*v),
        GgufKvValue::Int8(v) => buf.push(*v as u8),
        GgufKvValue::Uint16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufKvValue::Int16(v) => buf.extend_from_slice(&(*v as i16).to_le_bytes()),
        GgufKvValue::Uint32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufKvValue::Int32(v) => buf.extend_from_slice(&(*v as i32).to_le_bytes()),
        GgufKvValue::Uint64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufKvValue::Int64(v) => buf.extend_from_slice(&(*v as i64).to_le_bytes()),
        GgufKvValue::Float32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufKvValue::Float16(v) => buf.extend_from_slice(&(*v as u16).to_le_bytes()),
        GgufKvValue::Bool(v) => buf.push(*v as u8),
        GgufKvValue::String(s) => {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        GgufKvValue::Array(arr) => {
            // GGUF v3: element_type (u32), count (u64), then elements
            let element_type: u32 = 8; // GgufValueType::String
            buf.extend_from_slice(&element_type.to_le_bytes());
            buf.extend_from_slice(&(arr.len() as u64).to_le_bytes());
            for item in arr {
                match item {
                    GgufKvValue::String(s) => {
                        // Real GGUF files use u64 for string array element lengths
                        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
                        buf.extend_from_slice(s.as_bytes());
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

fn write_tensor_info_raw(buf: &mut Vec<u8>, name: &str, shape: &[u64], dtype: u32, offset: u64) {
    let name_bytes = name.as_bytes();
    buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for dim in shape {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    buf.extend_from_slice(&dtype.to_le_bytes());
    buf.extend_from_slice(&offset.to_le_bytes());
}

fn make_test_gguf(path: &PathBuf) {
    let vocab_size: u32 = 10;
    let dummy_tokens: Vec<GgufKvValue> = (0..vocab_size as usize)
        .map(|i| GgufKvValue::String(format!("tok{}", i)))
        .collect();

    let kv_pairs: Vec<GgufKvPair> = vec![
        kv_pair_str("general.architecture", "llama"),
        kv_pair_str("general.file_type", "F16"),
        kv_pair_u64("general.alignment", 32),
        kv_pair_u64("llama.context_length", 4096),
        kv_pair_u64("llama.embedding_length", 64),
        kv_pair_u64("llama.block_count", 1),
        kv_pair_u64("llama.attention.head_count", 4),
        kv_pair_u64("llama.attention.head_count_kv", 2),
        kv_pair_u64("llama.feed_forward_length", 128),
        kv_pair_u64("llama.rope.dimension_count", 64),
        kv_pair_f32("llama.attention.layer_norm_rms_epsilon", 1e-5),
        kv_pair_u64("tokenizer.ggml.model", 2),
        kv_pair_array("tokenizer.ggml.tokens", dummy_tokens),
    ];

    let tensor_specs: Vec<(&str, Vec<u64>)> = vec![
        ("tok_embeddings.weight", vec![64]),
        ("output.weight", vec![vocab_size as u64, 64]),
        ("layers.0.attention.wq.weight", vec![64, 64]),
        ("layers.0.attention.wk.weight", vec![64, 64]),
        ("layers.0.attention.wv.weight", vec![64, 64]),
        ("layers.0.attention.wo.weight", vec![64, 64]),
        ("layers.0.attention_norm.weight", vec![64]),
        ("layers.0.ffn_norm.weight", vec![64]),
        ("layers.0.feed_forward.w1.weight", vec![64, 128]),
        ("layers.0.feed_forward.w2.weight", vec![128, 64]),
        ("layers.0.feed_forward.w3.weight", vec![64, 128]),
        ("norm.weight", vec![64]),
    ];

    // Build header to compute alignment
    let mut hdr = Vec::new();
    hdr.extend_from_slice(b"GGUF");
    hdr.extend_from_slice(&3u32.to_le_bytes());
    let tensor_count = tensor_specs.len() as u64;
    let kv_count = kv_pairs.len() as u64;
    hdr.extend_from_slice(&tensor_count.to_le_bytes());
    hdr.extend_from_slice(&kv_count.to_le_bytes());
    for kv in &kv_pairs {
        let key_bytes = kv.key.as_bytes();
        hdr.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        hdr.extend_from_slice(key_bytes);
        hdr.extend_from_slice(&kv.value_type.to_u32().to_le_bytes());
        write_kv_value(&mut hdr, &kv.value);
    }
    for (name, shape) in &tensor_specs {
        write_tensor_info_raw(&mut hdr, name, shape, 1, 0);
    }

    let buf_size_before = hdr.len() as u64;
    let data_section_start = (buf_size_before + 31) & !31;

    let mut cumulative = 0u64;
    let tensor_infos: Vec<_> = tensor_specs
        .iter()
        .map(|(name, shape)| {
            let info = (name.to_string(), shape.clone(), cumulative);
            let elems: u64 = shape.iter().product();
            cumulative += elems * 2;
            info
        })
        .collect();

    // Write final file
    let mut buf = Vec::new();
    buf.extend_from_slice(b"GGUF");
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&(tensor_specs.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(kv_pairs.len() as u64).to_le_bytes());
    for kv in &kv_pairs {
        let key_bytes = kv.key.as_bytes();
        buf.extend_from_slice(&(key_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(&kv.value_type.to_u32().to_le_bytes());
        write_kv_value(&mut buf, &kv.value);
    }
    for (name, shape, offset) in &tensor_infos {
        write_tensor_info_raw(&mut buf, name, shape, 1, *offset);
    }

    let total: u64 = tensor_infos
        .iter()
        .map(|(_, shape, _)| shape.iter().product::<u64>() * 2)
        .sum();

    buf.resize((data_section_start + total) as usize, 0);
    for i in 0..total as usize {
        buf[data_section_start as usize + i] = if i % 2 == 0 { 0x00 } else { 0x3F };
    }

    std::fs::write(path, &buf).unwrap();
}

#[test]
fn test_dispatch_context_gpu_detection() {
    let ctx = DispatchContext::new();
    println!("Prefer GPU: {}", ctx.prefer_gpu());
    println!("GPU Available: {}", ctx.gpu_available());
    println!("Device Info: {}", ctx.device_info());
}

#[test]
fn test_linear_dispatch_accuracy() {
    let x = vec![1.0f32, 2.0f32];
    let weights_f16 = vec![
        f16::from_f32(1.0),
        f16::from_f32(0.5),
        f16::from_f32(0.5),
        f16::from_f32(1.0),
    ];
    let weights_f32 = vec![1.0f32, 0.5f32, 0.5f32, 1.0f32];
    let bias = Some(vec![0.1f32, 0.1f32]);

    let linear = LinearDispatch::new(weights_f16, weights_f32, bias, 2, 2);
    let ctx = DispatchContext::new();

    let result = linear.forward(&ctx, &x, 1).expect("Linear dispatch failed");
    println!("linear result: {:?}", result);

    assert!((result[0] - 2.1).abs() < 1e-4);
    assert!((result[1] - 2.6).abs() < 1e-4);
}

#[test]
fn test_dispatch_vs_cpu_output() {
    let dir = tempdir().unwrap();
    let gguf_path = dir.path().join("test.gguf");
    make_test_gguf(&gguf_path);

    let debug_path = std::path::PathBuf::from("/tmp/debug_test.gguf");
    if let Err(_) = std::fs::copy(&gguf_path, &debug_path) {
        println!("Saved test GGUF to {:?}", debug_path);
    }

    let weights = load_gguf_weights(&gguf_path).expect("Failed to load GGUF weights");
    println!("Loaded {} tensors", weights.tensors.len());

    let mut cpu_model = CpuModel::load_gguf(&gguf_path).expect("Failed to load GGUF");
    let mut dispatch_model = CpuModel::load_gguf(&gguf_path).expect("Failed to load GGUF");
    dispatch_model.enable_dispatch();

    let token: u32 = 0;
    let cpu_logits = cpu_model.decode(token).expect("CPU decode failed");
    dispatch_model.reset();

    let dispatch_hidden = dispatch_model.llama_model.embed(token, 0).expect("embed failed");
    let dispatch_hidden = dispatch_model.forward_with_dispatch(&dispatch_hidden, 0).expect("forward_with_dispatch failed");
    let dispatch_logits = dispatch_model.apply_output_head(&dispatch_hidden).expect("apply_output_head failed");

    assert_eq!(
        cpu_logits.len(),
        dispatch_logits.len(),
        "Logit vector length mismatch"
    );
    for (i, (cpu, dispatch)) in cpu_logits.iter().zip(dispatch_logits.iter()).enumerate() {
        let diff = (cpu - dispatch).abs();
        let tol: f32 = 1e-3_f32.max(cpu.abs() * 1e-4);
        assert!(
            diff < tol,
            "Logit mismatch at index {}: cpu={:.6} dispatch={:.6} diff={:.6} tol={:.6}",
            i, cpu, dispatch, diff, tol
        );
    }
}

#[test]
fn test_dispatch_conformance_real_model() {
    // Use conformance corpus model (Q4_K_M quantization)
    let model_path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    );
    
    if !model_path.exists() {
        println!("Skipping conformance test: conformance corpus not found");
        return;
    }

    println!("\n=== Conformance: dispatch vs CPU baseline (Q4_K_M) ===");
    println!("Model: {}", model_path.display());

    // Load CPU baseline
    let mut cpu_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");

    // Load dispatch model
    let mut dispatch_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");
    dispatch_model.enable_dispatch();

    assert!(
        dispatch_model.can_use_dispatch(),
        "Dispatch should be enabled"
    );

    // Test with token 0 (BOS)
    let token: u32 = 0;

    // CPU path
    let cpu_logits = cpu_model.decode(token).expect("CPU decode failed");

    // Dispatch path
    let dispatch_hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("Embed failed");
    let dispatch_hidden = dispatch_model
        .forward_with_dispatch(&dispatch_hidden, 0)
        .expect("Forward failed");
    let dispatch_logits = dispatch_model
        .apply_output_head(&dispatch_hidden)
        .expect("Output head failed");

    // Compare with tolerance (1e-2 accounts for f16 precision loss)
    if cpu_logits.len() != dispatch_logits.len() {
        panic!(
            "Length mismatch: CPU len={} vs Dispatch len={}",
            cpu_logits.len(),
            dispatch_logits.len()
        );
    }

    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;

    for (i, (a, b)) in cpu_logits.iter().zip(dispatch_logits.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("Max abs diff: {:.6e} at index {}", max_diff, max_idx);
    println!(
        "CPU[{}]={:.8}, Dispatch[{}]={:.8}",
        max_idx, cpu_logits[max_idx], max_idx, dispatch_logits[max_idx]
    );

    if max_diff > 1e-2 {
        panic!(
            "Conformance failed: max diff {:.6e} exceeds tolerance 1e-2",
            max_diff
        );
    }

    println!("✅ Q4_K_M CPU and dispatch outputs match within tolerance");
}

#[test]
fn test_dispatch_conformance_q8_0() {
    // Use conformance corpus model (Q8_0 quantization)
    let model_path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q8_0.gguf"
    );
    
    if !model_path.exists() {
        println!("Skipping conformance test: Q8_0 conformance corpus not found");
        return;
    }

    println!("\n=== Conformance: dispatch vs CPU baseline (Q8_0) ===");
    println!("Model: {}", model_path.display());

    // Load CPU baseline
    let mut cpu_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");

    // Load dispatch model
    let mut dispatch_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");
    dispatch_model.enable_dispatch();

    assert!(
        dispatch_model.can_use_dispatch(),
        "Dispatch should be enabled"
    );

    // Test with token 0 (BOS)
    let token: u32 = 0;

    // CPU path
    let cpu_logits = cpu_model.decode(token).expect("CPU decode failed");

    // Dispatch path
    let dispatch_hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("Embed failed");
    let dispatch_hidden = dispatch_model
        .forward_with_dispatch(&dispatch_hidden, 0)
        .expect("Forward failed");
    let dispatch_logits = dispatch_model
        .apply_output_head(&dispatch_hidden)
        .expect("Output head failed");

    // Compare with tolerance (1e-2 accounts for f16 precision loss)
    if cpu_logits.len() != dispatch_logits.len() {
        panic!(
            "Length mismatch: CPU len={} vs Dispatch len={}",
            cpu_logits.len(),
            dispatch_logits.len()
        );
    }

    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;

    for (i, (a, b)) in cpu_logits.iter().zip(dispatch_logits.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("Max abs diff: {:.6e} at index {}", max_diff, max_idx);
    println!(
        "CPU[{}]={:.8}, Dispatch[{}]={:.8}",
        max_idx, cpu_logits[max_idx], max_idx, dispatch_logits[max_idx]
    );

    if max_diff > 1e-2 {
        panic!(
            "Conformance failed: max diff {:.6e} exceeds tolerance 1e-2",
            max_diff
        );
    }

    println!("✅ Q8_0 CPU and dispatch outputs match within tolerance");
}

#[test]
fn test_dispatch_conformance_q2_k() {
    // Use conformance corpus model (Q2_K quantization)
    let model_path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q2_k.gguf"
    );
    
    if !model_path.exists() {
        println!("Skipping conformance test: Q2_K conformance corpus not found");
        return;
    }

    println!("\n=== Conformance: dispatch vs CPU baseline (Q2_K) ===");
    println!("Model: {}", model_path.display());

    // Load CPU baseline
    let mut cpu_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");

    // Load dispatch model
    let mut dispatch_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");
    dispatch_model.enable_dispatch();

    assert!(
        dispatch_model.can_use_dispatch(),
        "Dispatch should be enabled"
    );

    // Test with token 0 (BOS)
    let token: u32 = 0;

    // CPU path
    let cpu_logits = cpu_model.decode(token).expect("CPU decode failed");

    // Dispatch path
    let dispatch_hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("Embed failed");
    let dispatch_hidden = dispatch_model
        .forward_with_dispatch(&dispatch_hidden, 0)
        .expect("Forward failed");
    let dispatch_logits = dispatch_model
        .apply_output_head(&dispatch_hidden)
        .expect("Output head failed");

    // Compare with tolerance (1e-2 accounts for f16 precision loss)
    if cpu_logits.len() != dispatch_logits.len() {
        panic!(
            "Length mismatch: CPU len={} vs Dispatch len={}",
            cpu_logits.len(),
            dispatch_logits.len()
        );
    }

    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;

    for (i, (a, b)) in cpu_logits.iter().zip(dispatch_logits.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("Max abs diff: {:.6e} at index {}", max_diff, max_idx);
    println!(
        "CPU[{}]={:.8}, Dispatch[{}]={:.8}",
        max_idx, cpu_logits[max_idx], max_idx, dispatch_logits[max_idx]
    );

    if max_diff > 1e-2 {
        panic!(
            "Conformance failed: max diff {:.6e} exceeds tolerance 1e-2",
            max_diff
        );
    }

    println!("✅ Q2_K CPU and dispatch outputs match within tolerance");
}

#[test]
fn test_dispatch_conformance_q3_k() {
    // Use conformance corpus model (Q3_K quantization)
    let model_path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q3_k.gguf"
    );
    
    if !model_path.exists() {
        println!("Skipping conformance test: Q3_K conformance corpus not found");
        return;
    }

    println!("\n=== Conformance: dispatch vs CPU baseline (Q3_K) ===");
    println!("Model: {}", model_path.display());

    // Load CPU baseline
    let mut cpu_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");

    // Load dispatch model
    let mut dispatch_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");
    dispatch_model.enable_dispatch();

    assert!(
        dispatch_model.can_use_dispatch(),
        "Dispatch should be enabled"
    );

    // Test with token 0 (BOS)
    let token: u32 = 0;

    // CPU path
    let cpu_logits = cpu_model.decode(token).expect("CPU decode failed");

    // Dispatch path
    let dispatch_hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("Embed failed");
    let dispatch_hidden = dispatch_model
        .forward_with_dispatch(&dispatch_hidden, 0)
        .expect("Forward failed");
    let dispatch_logits = dispatch_model
        .apply_output_head(&dispatch_hidden)
        .expect("Output head failed");

    // Compare with tolerance (1e-2 accounts for f16 precision loss)
    if cpu_logits.len() != dispatch_logits.len() {
        panic!(
            "Length mismatch: CPU len={} vs Dispatch len={}",
            cpu_logits.len(),
            dispatch_logits.len()
        );
    }

    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;

    for (i, (a, b)) in cpu_logits.iter().zip(dispatch_logits.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("Max abs diff: {:.6e} at index {}", max_diff, max_idx);
    println!(
        "CPU[{}]={:.8}, Dispatch[{}]={:.8}",
        max_idx, cpu_logits[max_idx], max_idx, dispatch_logits[max_idx]
    );

    if max_diff > 1e-2 {
        panic!(
            "Conformance failed: max diff {:.6e} exceeds tolerance 1e-2",
            max_diff
        );
    }

    println!("✅ Q3_K CPU and dispatch outputs match within tolerance");
}

#[test]
fn test_dispatch_conformance_q4_0() {
    // Use conformance corpus model (Q4_0 quantization)
    let model_path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_0.gguf"
    );
    
    if !model_path.exists() {
        println!("Skipping conformance test: Q4_0 conformance corpus not found");
        return;
    }

    println!("\n=== Conformance: dispatch vs CPU baseline (Q4_0) ===");
    println!("Model: {}", model_path.display());

    // Load CPU baseline
    let mut cpu_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");

    // Load dispatch model
    let mut dispatch_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");
    dispatch_model.enable_dispatch();

    assert!(
        dispatch_model.can_use_dispatch(),
        "Dispatch should be enabled"
    );

    // Test with token 0 (BOS)
    let token: u32 = 0;

    // CPU path
    let cpu_logits = cpu_model.decode(token).expect("CPU decode failed");

    // Dispatch path
    let dispatch_hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("Embed failed");
    let dispatch_hidden = dispatch_model
        .forward_with_dispatch(&dispatch_hidden, 0)
        .expect("Forward failed");
    let dispatch_logits = dispatch_model
        .apply_output_head(&dispatch_hidden)
        .expect("Output head failed");

    // Compare with tolerance (1e-2 accounts for f16 precision loss)
    if cpu_logits.len() != dispatch_logits.len() {
        panic!(
            "Length mismatch: CPU len={} vs Dispatch len={}",
            cpu_logits.len(),
            dispatch_logits.len()
        );
    }

    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;

    for (i, (a, b)) in cpu_logits.iter().zip(dispatch_logits.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("Max abs diff: {:.6e} at index {}", max_diff, max_idx);
    println!(
        "CPU[{}]={:.8}, Dispatch[{}]={:.8}",
        max_idx, cpu_logits[max_idx], max_idx, dispatch_logits[max_idx]
    );

    if max_diff > 1e-2 {
        panic!(
            "Conformance failed: max diff {:.6e} exceeds tolerance 1e-2",
            max_diff
        );
    }

    println!("✅ Q4_0 CPU and dispatch outputs match within tolerance");
}

#[test]
fn test_dispatch_conformance_q5_k() {
    // Use conformance corpus model (Q5_K quantization)
    let model_path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q5_k.gguf"
    );
    
    if !model_path.exists() {
        println!("Skipping conformance test: Q5_K conformance corpus not found");
        return;
    }

    println!("\n=== Conformance: dispatch vs CPU baseline (Q5_K) ===");
    println!("Model: {}", model_path.display());

    // Load CPU baseline
    let mut cpu_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");

    // Load dispatch model
    let mut dispatch_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");
    dispatch_model.enable_dispatch();

    assert!(
        dispatch_model.can_use_dispatch(),
        "Dispatch should be enabled"
    );

    // Test with token 0 (BOS)
    let token: u32 = 0;

    // CPU path
    let cpu_logits = cpu_model.decode(token).expect("CPU decode failed");

    // Dispatch path
    let dispatch_hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("Embed failed");
    let dispatch_hidden = dispatch_model
        .forward_with_dispatch(&dispatch_hidden, 0)
        .expect("Forward failed");
    let dispatch_logits = dispatch_model
        .apply_output_head(&dispatch_hidden)
        .expect("Output head failed");

    // Compare with tolerance (1e-2 accounts for f16 precision loss)
    if cpu_logits.len() != dispatch_logits.len() {
        panic!(
            "Length mismatch: CPU len={} vs Dispatch len={}",
            cpu_logits.len(),
            dispatch_logits.len()
        );
    }

    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;

    for (i, (a, b)) in cpu_logits.iter().zip(dispatch_logits.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("Max abs diff: {:.6e} at index {}", max_diff, max_idx);
    println!(
        "CPU[{}]={:.8}, Dispatch[{}]={:.8}",
        max_idx, cpu_logits[max_idx], max_idx, dispatch_logits[max_idx]
    );

    if max_diff > 1e-2 {
        panic!(
            "Conformance failed: max diff {:.6e} exceeds tolerance 1e-2",
            max_diff
        );
    }

    println!("✅ Q5_K CPU and dispatch outputs match within tolerance");
}

#[test]
fn test_dispatch_conformance_q6_k() {
    // Use conformance corpus model (Q6_K quantization)
    let model_path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q6_k.gguf"
    );
    
    if !model_path.exists() {
        println!("Skipping conformance test: Q6_K conformance corpus not found");
        return;
    }

    println!("\n=== Conformance: dispatch vs CPU baseline (Q6_K) ===");
    println!("Model: {}", model_path.display());

    // Load CPU baseline
    let mut cpu_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");

    // Load dispatch model
    let mut dispatch_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");
    dispatch_model.enable_dispatch();

    assert!(
        dispatch_model.can_use_dispatch(),
        "Dispatch should be enabled"
    );

    // Test with token 0 (BOS)
    let token: u32 = 0;

    // CPU path
    let cpu_logits = cpu_model.decode(token).expect("CPU decode failed");

    // Dispatch path
    let dispatch_hidden = dispatch_model
        .llama_model
        .embed(token, 0)
        .expect("Embed failed");
    let dispatch_hidden = dispatch_model
        .forward_with_dispatch(&dispatch_hidden, 0)
        .expect("Forward failed");
    let dispatch_logits = dispatch_model
        .apply_output_head(&dispatch_hidden)
        .expect("Output head failed");

    // Compare with tolerance (1e-2 accounts for f16 precision loss)
    if cpu_logits.len() != dispatch_logits.len() {
        panic!(
            "Length mismatch: CPU len={} vs Dispatch len={}",
            cpu_logits.len(),
            dispatch_logits.len()
        );
    }

    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;

    for (i, (a, b)) in cpu_logits.iter().zip(dispatch_logits.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("Max abs diff: {:.6e} at index {}", max_diff, max_idx);
    println!(
        "CPU[{}]={:.8}, Dispatch[{}]={:.8}",
        max_idx, cpu_logits[max_idx], max_idx, dispatch_logits[max_idx]
    );

    if max_diff > 1e-2 {
        panic!(
            "Conformance failed: max diff {:.6e} exceeds tolerance 1e-2",
            max_diff
        );
    }

    println!("✅ Q6_K CPU and dispatch outputs match within tolerance");
}

#[test]
fn test_dispatch_cpu_fallback() {
    let dir = tempdir().unwrap();
    let gguf_path = dir.path().join("test.gguf");
    make_test_gguf(&gguf_path);

    let mut model = CpuModel::load_gguf(&gguf_path).expect("Failed to load GGUF");
    model.enable_dispatch();

    let hidden = model.llama_model.embed(0, 0).expect("embed failed");
    let result = model
        .forward_with_dispatch(&hidden, 0)
        .expect("dispatch should fall back to CPU");
    assert_eq!(result.len(), hidden.len(), "Dispatch output shape mismatch");
    println!("Dispatch fallback output shape: {}", result.len());
}

#[test]
#[ignore = "Requires conformance-corpus/qwen2.5-0.5b-instruct-f16.gguf"]
fn test_dispatch_conformance_f16_model() {
    use pesti_runner::model::CpuModel;
    
    let model_path = std::path::Path::new(
        "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-f16.gguf"
    );
    
    if !model_path.exists() {
        println!("Skipping conformance test: F16 model not found");
        return;
    }

    println!(
        "
=== Conformance: dispatch vs CPU baseline (F16 model) ==="
    );
    println!("Model: {}", model_path.display());

    // Load CPU baseline
    let mut cpu_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");

    // Load dispatch model
    let mut dispatch_model = CpuModel::load_gguf(model_path).expect("Failed to load GGUF");
    dispatch_model.enable_dispatch();

    assert!(
        dispatch_model.can_use_dispatch(),
        "Dispatch should be enabled"
    );

    // Test with token 0 (BOS)
    let token: u32 = 0;

    // CPU path
    let cpu_logits = cpu_model.decode(token).expect("CPU decode failed");
    println!("CPU logits shape: {}", cpu_logits.len());

    // Dispatch path
    let dispatch_hidden = dispatch_model.llama_model.embed(token, 0).expect("Embed failed");
    let dispatch_hidden = dispatch_model.forward_with_dispatch(&dispatch_hidden, 0).expect("Forward failed");
    let dispatch_logits = dispatch_model.apply_output_head(&dispatch_hidden).expect("Output head failed");
    println!("Dispatch logits shape: {}", dispatch_logits.len());

    // Compare with tolerance (1e-2 accounts for f16 precision loss)
    if cpu_logits.len() != dispatch_logits.len() {
        panic!(
            "Length mismatch: CPU len={} vs Dispatch len={}",
            cpu_logits.len(), dispatch_logits.len()
        );
    }

    let mut max_diff = 0.0f32;
    let mut max_idx = 0usize;

    for (i, (a, b)) in cpu_logits.iter().zip(dispatch_logits.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    println!("Max abs diff: {:.6e} at index {}", max_diff, max_idx);
    println!(
        "CPU[{}]={:.8}, Dispatch[{}]={:.8}",
        max_idx, cpu_logits[max_idx], max_idx, dispatch_logits[max_idx]
    );

    if max_diff > 1e-2 {
        panic!("Conformance failed: max diff {:.6e} exceeds tolerance 1e-2", max_diff);
    }

    println!("✅ CPU and dispatch outputs match within tolerance");
}
