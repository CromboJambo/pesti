//! Optimized CPU reference implementation with SIMD and parallelism

use half::f16;
use rayon::prelude::*;

/// Apply RoPE using vectorized operations
fn apply_rope_cpu_optimized(q: &mut [f32], head_dim: usize, pos: usize, rope_base: f32) {
    let half_dim = head_dim / 2;
    
    for dim_pair in 0..half_dim {
        let idx = dim_pair * 2;
        if idx + 1 >= q.len() { 
            continue; 
        }
        
        // Precomputed frequency for this dimension pair
        let inv_freq = 1.0 / (rope_base.powf((dim_pair as f32) / half_dim as f32));
        let freq = pos as f32 * inv_freq;
        
        // Vectorized trig: compute cos/sin once per dimension pair
        let cos_val = freq.cos();
        let sin_val = freq.sin();
        
        let q0 = q[idx];
        let q1 = q[idx + 1];
        
        // RoPE rotation (can be SIMD'd as 2-element vectors)
        q[idx] = q0 * cos_val - q1 * sin_val;
        q[idx + 1] = q0 * sin_val + q1 * cos_val;
    }
}

/// Optimized dot product using manual SIMD-like unrolling
#[inline]
fn simd_dot_product(q: &[f32], k: &[f32], head_dim: usize) -> f32 {
    // Unroll by 4 for better vectorization potential
    let mut sum = 0.0f32;
    let chunk_size = 4;
    
    for chunk in (0..head_dim).step_by(chunk_size) {
        if chunk + 3 < head_dim {
            // Process 4 elements at once (SIMD-friendly)
            unsafe {
                let q_ptr = q.as_ptr().add(chunk);
                let k_ptr = k.as_ptr().add(chunk);
                
                // Manual unrolling - compiler should vectorize this
                sum += *q_ptr.offset(0) * *k_ptr.offset(0);
                sum += *q_ptr.offset(1) * *k_ptr.offset(1);
                sum += *q_ptr.offset(2) * *k_ptr.offset(2);
                sum += *q_ptr.offset(3) * *k_ptr.offset(3);
            }
        } else {
            // Handle remaining elements
            for i in chunk..head_dim {
                sum += q[i] * k[i];
            }
        }
    }
    
    sum
}

/// Optimized softmax with numerical stability
#[inline]
fn optimized_softmax(scores: &[f32]) -> Vec<f32> {
    let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Compute exp in parallel
    let exps: Vec<f32> = scores.par_iter()
        .map(|&s| (s - max_val).exp())
        .collect();
    
    // Parallel sum
    let sum: f32 = exps.par_iter().sum();
    
    if sum > 0.0 {
        exps.iter().map(|&e| e / sum).collect()
    } else {
        vec![1.0 / scores.len() as f32; scores.len()]
    }
}

/// Reference implementation with optimizations
pub fn reference_raw_scores_optimized(
    q_h: &[f16],
    k_h: &[f16],
    v_h: &[f16],
    seq_q: usize, 
    seq_k: usize, 
    num_heads: usize, 
    head_dim: usize,
    rope_base: f32, 
    scale: f32,
) -> Vec<f32> {
    let mut q_rope = vec![0.0f32; seq_q * num_heads * head_dim];
    let mut k_rope = vec![0.0f32; seq_k * num_heads * head_dim];
    
    // Convert to f32 in parallel
    q_rope.par_iter_mut()
        .zip(q_h.par_iter())
        .for_each(|(q_out, &q_in)| *q_out = q_in.to_f32());
    
    k_rope.par_iter_mut()
        .zip(k_h.par_iter())
        .for_each(|(k_out, &k_in)| *k_out = k_in.to_f32());
    
    // Apply RoPE in parallel across heads
    for pos in 0..seq_q {
        q_rope.par_chunks_mut(head_dim)
            .enumerate()
            .filter(|(_, chunk)| chunk.len() == head_dim)
            .for_each(|(h, chunk)| {
                let _start_idx = h * head_dim;
                // Apply RoPE to each head independently (can be parallelized further)
                for dim_pair in 0..head_dim / 2 {
                    let idx = dim_pair * 2;
                    if idx + 1 < chunk.len() {
                        let inv_freq = 1.0 / (rope_base.powf((dim_pair as f32) / (head_dim / 2) as f32));
                        let freq = pos as f32 * inv_freq;
                        let cos_val = freq.cos();
                        let sin_val = freq.sin();
                        
                        let q0 = chunk[idx];
                        let q1 = chunk[idx + 1];
                        chunk[idx] = q0 * cos_val - q1 * sin_val;
                        chunk[idx + 1] = q0 * sin_val + q1 * cos_val;
                    }
                }
            });
    }
    
    for pos in 0..seq_k {
        k_rope.par_chunks_mut(head_dim)
            .enumerate()
            .filter(|(_, chunk)| chunk.len() == head_dim)
            .for_each(|(_h, chunk)| {
                for dim_pair in 0..head_dim / 2 {
                    let idx = dim_pair * 2;
                    if idx + 1 < chunk.len() {
                        let inv_freq = 1.0 / (rope_base.powf((dim_pair as f32) / (head_dim / 2) as f32));
                        let freq = pos as f32 * inv_freq;
                        let cos_val = freq.cos();
                        let sin_val = freq.sin();
                        
                        let k0 = chunk[idx];
                        let k1 = chunk[idx + 1];
                        chunk[idx] = k0 * cos_val - k1 * sin_val;
                        chunk[idx + 1] = k0 * sin_val + k1 * cos_val;
                    }
                }
            });
    }
    
    // Compute attention output in parallel across query positions and heads
    let mut output = vec![0.0f32; seq_q * num_heads * head_dim];
    
    // Parallelize over (q_pos, head) pairs
    for q_pos in 0..seq_q {
        for h in 0..num_heads {
            let q_head = &q_rope[q_pos * num_heads * head_dim + h * head_dim..][..head_dim];
            
            // Compute all scores in parallel
            let scores: Vec<f32> = (0..seq_k)
                .into_par_iter()
                .map(|k_pos| {
                    let k_head = &k_rope[k_pos * num_heads * head_dim + h * head_dim..][..head_dim];
                    let dot = simd_dot_product(q_head, k_head, head_dim);
                    
                    let score = dot * scale;
                    if k_pos > q_pos {
                        -1e9 // Causal mask
                    } else {
                        score
                    }
                })
                .collect();
            
            // Softmax
            let weights = optimized_softmax(&scores);
            
            // Weighted sum of V (parallelize over dimensions)
            let attn_output: Vec<f32> = (0..head_dim)
                .into_par_iter()
                .map(|d| {
                    let mut sum = 0.0f32;
                    for k_pos in 0..seq_k {
                        let v_idx = k_pos * num_heads * head_dim + h * head_dim + d;
                        let v_val = v_h[v_idx].to_f32();
                        sum += weights[k_pos] * v_val;
                    }
                    sum
                })
                .collect();
            
            // Write output
            for (d, &val) in attn_output.iter().enumerate() {
                output[q_pos * num_heads * head_dim + h * head_dim + d] = val;
            }
        }
    }
    
    output
}

/// Alternative using gemm crate for matrix multiplication
#[cfg(feature = "gemm")]
pub fn reference_with_gemm(
    q_h: &[f16],
    k_h: &[f16],
    v_h: &[f16],
    seq_q: usize, 
    seq_k: usize, 
    num_heads: usize, 
    head_dim: usize,
    rope_base: f32, 
    scale: f32,
) -> Vec<f32> {
    use gemm::f32::{Gemm, C};
    
    // Convert to f32 matrices per head
    let mut output = vec![0.0f32; seq_q * num_heads * head_dim];
    
    for h in 0..num_heads {
        // Build Q matrix: [seq_q, head_dim]
        let q_mat: Vec<f32> = (0..seq_q)
            .flat_map(|q_pos| {
                (0..head_dim).map(|d| {
                    let idx = q_pos * num_heads * head_dim + h * head_dim + d;
                    q_h[idx].to_f32()
                })
            })
            .collect();
        
        // Build K matrix: [seq_k, head_dim] (transpose for GEMM)
        let k_mat: Vec<f32> = (0..seq_k)
            .flat_map(|k_pos| {
                (0..head_dim).map(|d| {
                    let idx = k_pos * num_heads * head_dim + h * head_dim + d;
                    k_h[idx].to_f32()
                })
            })
            .collect();
        
        // Build V matrix: [seq_k, head_dim]
        let v_mat: Vec<f32> = (0..seq_k)
            .flat_map(|k_pos| {
                (0..head_dim).map(|d| {
                    let idx = k_pos * num_heads * head_dim + h * head_dim + d;
                    v_h[idx].to_f32()
                })
            })
            .collect();
        
        // Compute Q @ K^T using gemm (result: [seq_q, seq_k])
        let scores_mat = Gemm::new(C(0.0), &q_mat, &k_mat);
        
        // Apply causal mask and softmax per row
        let mut weights_mat = vec![0.0f32; seq_q * seq_k];
        for q_pos in 0..seq_q {
            let max_val = (0..seq_k)
                .map(|k_pos| scores_mat[q_pos * seq_k + k_pos])
                .fold(f32::NEG_INFINITY, f32::max);
            
            let sum: f32 = (0..seq_k)
                .map(|k_pos| {
                    let score = scores_mat[q_pos * seq_k + k_pos];
                    if k_pos > q_pos {
                        0.0 // Will be masked out
                    } else {
                        (score * scale - max_val).exp()
                    }
                })
                .sum();
            
            for k_pos in 0..seq_k {
                let score = scores_mat[q_pos * seq_k + k_pos];
                if k_pos > q_pos {
                    weights_mat[q_pos * seq_k + k_pos] = 0.0;
                } else if sum > 0.0 {
                    weights_mat[q_pos * seq_k + k_pos] = (score * scale - max_val).exp() / sum;
                } else {
                    weights_mat[q_pos * seq_k + k_pos] = 1.0 / seq_k as f32;
                }
            }
        }
        
        // Compute weights @ V: [seq_q, head_dim]
        let attn_output = Gemm::new(C(1.0), &weights_mat, &v_mat);
        
        // Write to output buffer
        for q_pos in 0..seq_q {
            for d in 0..head_dim {
                let idx = q_pos * num_heads * head_dim + h * head_dim + d;
                output[idx] = attn_output[q_pos * head_dim + d];
            }
        }
    }
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_dot_product() {
        let q: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let k: Vec<f32> = vec![1.0; 16];
        
        let result = simd_dot_product(&q, &k, 16);
        let expected: f32 = (0..16).map(|x| x as f32).sum();
        
        assert!((result - expected).abs() < 1e-5);
    }
    
    #[test]
    fn test_optimized_softmax() {
        let scores = vec![3.0, 4.0, 2.0];
        let weights = optimized_softmax(&scores);
        
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
