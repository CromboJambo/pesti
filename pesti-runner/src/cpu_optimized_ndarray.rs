//! CPU reference implementation using ndarray for structured array operations

use half::f16;
use ndarray::{Array2, ArrayView1};
use rayon::prelude::*;

/// Apply RoPE with ndarray-based structure
fn apply_rope_ndarray(q: &mut [f32], head_dim: usize, pos: usize, rope_base: f32) {
    let half_dim = head_dim / 2;
    for dim_pair in 0..half_dim {
        let idx = dim_pair * 2;
        if idx + 1 >= q.len() {
            continue;
        }

        // Precomputed frequency for this dimension pair
        let inv_freq = 1.0 / (rope_base.powf((dim_pair as f32) / half_dim as f32));
        let freq = pos as f32 * inv_freq;

        let cos_val = freq.cos();
        let sin_val = freq.sin();

        let q0 = q[idx];
        let q1 = q[idx + 1];

        // RoPE rotation
        q[idx] = q0 * cos_val - q1 * sin_val;
        q[idx + 1] = q0 * sin_val + q1 * cos_val;
    }
}

/// Reference implementation using ndarray for structured array operations
pub fn reference_with_ndarray(
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
    // Parallelize over heads and collect results
    let head_results: Vec<Vec<f32>> = (0..num_heads)
        .into_par_iter()
        .map(|h| {
            // Build Q matrix: [seq_q, head_dim]
            let q_mat: Array2<f32> = Array2::from_shape_fn((seq_q, head_dim), |(q_pos, d)| {
                f16::to_f32(q_h[q_pos * num_heads * head_dim + h * head_dim + d])
            });

            // Build K matrix: [seq_k, head_dim]
            let k_mat: Array2<f32> = Array2::from_shape_fn((seq_k, head_dim), |(k_pos, d)| {
                f16::to_f32(k_h[k_pos * num_heads * head_dim + h * head_dim + d])
            });

            // Build V matrix: [seq_k, head_dim]
            let v_mat: Array2<f32> = Array2::from_shape_fn((seq_k, head_dim), |(k_pos, d)| {
                f16::to_f32(v_h[k_pos * num_heads * head_dim + h * head_dim + d])
            });

            // Apply RoPE to Q and K (sequential per head)
            let mut q_rope = q_mat.clone();
            let mut k_rope = k_mat.clone();

            for q_pos in 0..seq_q {
                let mut row_vec = [0.0f32; 128]; // Max head_dim we'll support
                for d in 0..head_dim {
                    row_vec[d] = q_rope[[q_pos, d]];
                }
                apply_rope_ndarray(&mut row_vec[..head_dim], head_dim, q_pos, rope_base);
                for d in 0..head_dim {
                    q_rope[[q_pos, d]] = row_vec[d];
                }
            }

            for k_pos in 0..seq_k {
                let mut row_vec = [0.0f32; 128]; // Max head_dim we'll support
                for d in 0..head_dim {
                    row_vec[d] = k_rope[[k_pos, d]];
                }
                apply_rope_ndarray(&mut row_vec[..head_dim], head_dim, k_pos, rope_base);
                for d in 0..head_dim {
                    k_rope[[k_pos, d]] = row_vec[d];
                }
            }

            // Compute Q @ K^T using ndarray (optimized, may use SIMD internally)
            let scores = q_rope.dot(&k_rope.t()); // [seq_q, seq_k]

            // Apply causal mask and softmax per row
            let mut weights_mat = Array2::zeros((seq_q, seq_k));
            for q_pos in 0..seq_q {
                let score_row = scores.row(q_pos);

                // Find max for numerical stability
                let max_val = score_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                // Compute softmax with causal mask
                let sum: f32 = (0..seq_k)
                    .map(|k_pos| {
                        let score = score_row[k_pos];
                        if k_pos > q_pos {
                            0.0
                        } else {
                            (score * scale - max_val).exp()
                        }
                    })
                    .sum();

                for k_pos in 0..seq_k {
                    let score = scores[(q_pos, k_pos)];
                    if k_pos > q_pos {
                        weights_mat[[q_pos, k_pos]] = 0.0;
                    } else {
                        weights_mat[[q_pos, k_pos]] = (score * scale - max_val).exp() / sum;
                    }
                }
            }

            // Compute weights @ V: [seq_q, head_dim] using ndarray GEMM
            let attn_output = weights_mat.dot(&v_mat);

            // Collect output for this head
            let mut head_output = vec![0.0f32; seq_q * head_dim];
            for q_pos in 0..seq_q {
                for d in 0..head_dim {
                    head_output[q_pos * head_dim + d] = attn_output[[q_pos, d]];
                }
            }

            head_output
        })
        .collect();

    // Combine all head results
    let mut output = Vec::with_capacity(seq_q * num_heads * head_dim);
    for h in 0..num_heads {
        let start_idx = h * seq_q * head_dim;
        for i in 0..seq_q * head_dim {
            output.push(head_results[h][i]);
        }
    }

    output
}

/// SIMD-friendly dot product with manual unrolling
fn simd_dot_product_ndarray(q: &ArrayView1<f32>, k: &ArrayView1<f32>) -> f32 {
    let head_dim = q.len();
    let mut sum = 0.0f32;

    // Unroll by 4 for SIMD-friendly access
    for chunk in (0..head_dim).step_by(4) {
        if chunk + 3 < head_dim {
            unsafe {
                let q_ptr = q.as_ptr().add(chunk);
                let k_ptr = k.as_ptr().add(chunk);

                sum += *q_ptr.offset(0) * *k_ptr.offset(0);
                sum += *q_ptr.offset(1) * *k_ptr.offset(1);
                sum += *q_ptr.offset(2) * *k_ptr.offset(2);
                sum += *q_ptr.offset(3) * *k_ptr.offset(3);
            }
        } else {
            for i in chunk..head_dim {
                sum += q[i] * k[i];
            }
        }
    }

    sum
}

/// Reference implementation using ndarray + manual dot products (for comparison)
pub fn reference_with_ndarray_manual(
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
    // Parallelize over heads and collect results
    let head_results: Vec<Vec<f32>> = (0..num_heads)
        .into_par_iter()
        .map(|h| {
            // Build Q and K matrices
            let q_mat: Array2<f32> = Array2::from_shape_fn((seq_q, head_dim), |(q_pos, d)| {
                f16::to_f32(q_h[q_pos * num_heads * head_dim + h * head_dim + d])
            });

            let k_mat: Array2<f32> = Array2::from_shape_fn((seq_k, head_dim), |(k_pos, d)| {
                f16::to_f32(k_h[k_pos * num_heads * head_dim + h * head_dim + d])
            });

            // Build V matrix
            let v_mat: Array2<f32> = Array2::from_shape_fn((seq_k, head_dim), |(k_pos, d)| {
                f16::to_f32(v_h[k_pos * num_heads * head_dim + h * head_dim + d])
            });

            // Apply RoPE
            let mut q_rope = q_mat.clone();
            let mut k_rope = k_mat.clone();

            for q_pos in 0..seq_q {
                let mut row_vec = [0.0f32; 128]; // Max head_dim we'll support
                for d in 0..head_dim {
                    row_vec[d] = q_rope[[q_pos, d]];
                }
                apply_rope_ndarray(&mut row_vec[..head_dim], head_dim, q_pos, rope_base);
                for d in 0..head_dim {
                    q_rope[[q_pos, d]] = row_vec[d];
                }
            }

            for k_pos in 0..seq_k {
                let mut row_vec = [0.0f32; 128]; // Max head_dim we'll support
                for d in 0..head_dim {
                    row_vec[d] = k_rope[[k_pos, d]];
                }
                apply_rope_ndarray(&mut row_vec[..head_dim], head_dim, k_pos, rope_base);
                for d in 0..head_dim {
                    k_rope[[k_pos, d]] = row_vec[d];
                }
            }

            // Compute scores using manual dot product (with SIMD unrolling)
            let mut scores = Array2::zeros((seq_q, seq_k));
            for q_pos in 0..seq_q {
                for k_pos in 0..seq_k {
                    let q_row = q_rope.row(q_pos);
                    let k_row = k_rope.row(k_pos);

                    let dot = simd_dot_product_ndarray(&q_row, &k_row);
                    scores[[q_pos, k_pos]] = if k_pos > q_pos { -1e9 } else { dot * scale };
                }
            }

            // Softmax and V multiplication (same as above)
            let mut weights_mat = Array2::zeros((seq_q, seq_k));
            for q_pos in 0..seq_q {
                let score_row = scores.row(q_pos);
                let max_val = score_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                let sum: f32 = (0..seq_k)
                    .map(|k_pos| {
                        let score = score_row[k_pos];
                        if k_pos > q_pos {
                            0.0
                        } else {
                            (score * scale - max_val).exp()
                        }
                    })
                    .sum();

                for k_pos in 0..seq_k {
                    let score = scores[(q_pos, k_pos)];
                    if k_pos > q_pos {
                        weights_mat[[q_pos, k_pos]] = 0.0;
                    } else {
                        weights_mat[[q_pos, k_pos]] = (score * scale - max_val).exp() / sum;
                    }
                }
            }

            let attn_output = weights_mat.dot(&v_mat);

            // Collect output for this head
            let mut head_output = vec![0.0f32; seq_q * head_dim];
            for q_pos in 0..seq_q {
                for d in 0..head_dim {
                    head_output[q_pos * head_dim + d] = attn_output[[q_pos, d]];
                }
            }

            head_output
        })
        .collect();

    // Combine all head results
    let mut output = Vec::with_capacity(seq_q * num_heads * head_dim);
    for h in 0..num_heads {
        let start_idx = h * seq_q * head_dim;
        for i in 0..seq_q * head_dim {
            output.push(head_results[h][i]);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndarray_basic() {
        let seq_q = 2;
        let seq_k = 3;
        let num_heads = 1;
        let head_dim = 4;

        let q_h: Vec<f16> = (0..seq_q * num_heads * head_dim)
            .map(|i| f16::from_f32(i as f32))
            .collect();

        let k_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
            .map(|i| f16::from_f32(i as f32))
            .collect();

        let v_h: Vec<f16> = (0..seq_k * num_heads * head_dim)
            .map(|i| f16::from_f32(i as f32))
            .collect();

        let rope_base = 10_000.0;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let result = reference_with_ndarray(
            &q_h, &k_h, &v_h, seq_q, seq_k, num_heads, head_dim, rope_base, scale,
        );

        assert_eq!(result.len(), seq_q * num_heads * head_dim);
        println!("Result: {:?}", result);
    }
}
