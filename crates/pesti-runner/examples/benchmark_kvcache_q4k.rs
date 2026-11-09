//! Benchmark: KV cache memory bandwidth comparison (FP16 vs Q4_K)
//!
//! Measures the theoretical memory bandwidth savings from Q4_K quantized
//! KV cache during autoregressive decode.

use pesti_runner::kernel::{Kvcache, Q4KVCache};

fn main() {
    // Qwen2.5-0.5B-Instruct config: 8 layers, 8 heads, head_dim=64
    let num_layers = 8;
    let num_kv_heads = 8;
    let head_dim = 64;
    let max_seq = 2048;

    println!("=== KV Cache Memory Bandwidth Benchmark ===");
    println!();
    println!("Model: Qwen2.5-0.5B-Instruct");
    println!("Layers: {}", num_layers);
    println!("KV heads: {}", num_kv_heads);
    println!("Head dim: {}", head_dim);
    println!("Max seq: {}", max_seq);
    println!();

    // FP16 KV cache (current)
    let fp16_cache = Kvcache::new(num_kv_heads, num_kv_heads, head_dim, max_seq, false);
    let fp16_bytes_per_layer = fp16_cache.total_elements() * 2; // f16 = 2 bytes

    // Q4_K KV cache (optimized)
    let q4k_cache = Q4KVCache::new(num_kv_heads, head_dim, max_seq);
    let q4k_bytes_per_layer = q4k_cache.memory_bytes();

    println!("Per-layer memory usage:");
    println!(
        "  FP16 KV cache: {} bytes ({:.2} KB)",
        fp16_bytes_per_layer,
        fp16_bytes_per_layer as f64 / 1024.0
    );
    println!(
        "  Q4_K KV cache: {} bytes ({:.2} KB)",
        q4k_bytes_per_layer,
        q4k_bytes_per_layer as f64 / 1024.0
    );
    println!();

    let total_fp16 = fp16_bytes_per_layer * num_layers;
    let total_q4k = q4k_bytes_per_layer * num_layers;

    println!("Total model KV cache ({} layers):", num_layers);
    println!(
        "  FP16: {} bytes ({:.2} MB)",
        total_fp16,
        total_fp16 as f64 / (1024.0 * 1024.0)
    );
    println!(
        "  Q4_K: {} bytes ({:.2} MB)",
        total_q4k,
        total_q4k as f64 / (1024.0 * 1024.0)
    );
    println!();

    let savings = ((total_fp16 as f64 - total_q4k as f64) / total_fp16 as f64) * 100.0;
    println!(
        "Memory savings: {:.1}% ({:.2} MB saved)",
        savings,
        (total_fp16 - total_q4k) as f64 / (1024.0 * 1024.0)
    );
    println!();

    // Theoretical bandwidth impact during autoregressive decode
    // Each token generation reads entire KV cache for attention computation
    let seq_len = 512; // Typical decode sequence length
    let kv_read_bytes_per_token_fp16 = (total_fp16 as u64) * (seq_len as u64) / (max_seq as u64);
    let kv_read_bytes_per_token_q4k = (total_q4k as u64) * (seq_len as u64) / (max_seq as u64);

    println!(
        "Memory bandwidth during decode (seq={}, 1 token generated):",
        seq_len
    );
    println!(
        "  FP16: {} bytes ({:.2} KB)",
        kv_read_bytes_per_token_fp16,
        kv_read_bytes_per_token_fp16 as f64 / 1024.0
    );
    println!(
        "  Q4_K: {} bytes ({:.2} KB)",
        kv_read_bytes_per_token_q4k,
        kv_read_bytes_per_token_q4k as f64 / 1024.0
    );

    let bw_savings = ((kv_read_bytes_per_token_fp16 as f64 - kv_read_bytes_per_token_q4k as f64)
        / kv_read_bytes_per_token_fp16 as f64)
        * 100.0;
    println!("  Bandwidth savings: {:.1}%", bw_savings);
    println!();

    // Estimated throughput improvement (memory-bound operation)
    let estimated_speedup =
        kv_read_bytes_per_token_fp16 as f64 / kv_read_bytes_per_token_q4k as f64;
    println!(
        "Estimated attention kernel speedup (if memory-bound): {:.2}x",
        estimated_speedup
    );
    println!("Note: Actual speedup depends on compute vs memory balance in kernel.");
}
