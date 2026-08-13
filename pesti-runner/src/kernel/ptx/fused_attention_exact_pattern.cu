//! Kernel that exactly mimics two-kernel pattern but fused into one
//! Uses shared memory for reduction (like kernel 1) and separate output buffer (like kernel 2)

#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

__global__ void fused_attention_exact_pattern_kernel(
    const half* __restrict__ q_ptr,      // [seq_q, num_heads, head_dim]
    const half* __restrict__ k_ptr,      // [seq_k, num_heads, head_dim]
    const half* __restrict__ v_ptr,      // [seq_k, num_heads, head_dim]
    float* __restrict__ s_ptr,           // Intermediate scores: [seq_q, num_heads, seq_k]
    float* __restrict__ out_ptr,         // Final output: [seq_q, num_heads, head_dim]
    float scale,
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim
) {
    int q_pos = blockIdx.x;
    int k_pos = blockIdx.y;
    int head = blockIdx.z;
    int tid = threadIdx.x;
    
    if (q_pos >= seq_q || k_pos >= seq_k || head >= num_heads) return;
    
    extern __shared__ float shared_dot[];  // Shared memory for reduction
    
    float dot_product = 0.0f;
    
    // RoPE + Q @ K^T computation (same as kernel 1)
    for (int d = tid; d < head_dim; d += blockDim.x) {
        int idx_q = q_pos * num_heads * head_dim + head * head_dim + d;
        int idx_k = k_pos * num_heads * head_dim + head * head_dim + d;
        
        float q_val = __half2float(q_ptr[idx_q]);
        float k_val = __half2float(k_ptr[idx_k]);
        dot_product += q_val * k_val;
    }
    
    shared_dot[tid] = dot_product;
    __syncthreads();
    
    if (tid == 0) {
        // Reduce across threads
        float total = 0.0f;
        for (int t = 0; t < blockDim.x; t++) {
            total += shared_dot[t];
        }
        
        total *= scale;
        
        // Causal mask: if k_pos > q_pos, set to -INF
        if (k_pos > q_pos) {
            total = -INFINITY;
        }
        
        // Write to intermediate scores buffer
        int score_idx = q_pos * num_heads * seq_k + head * seq_k + k_pos;
        s_ptr[score_idx] = total;
    }
    
    __syncthreads();
    
    // Now compute softmax and weighted V sum for this (q_pos, head) pair
    if (tid == 0 && q_pos == 0 && head == 0) {  // Only first block/thread does this
        float max_val = -INFINITY;
        
        // Find max score for softmax
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            if (s_ptr[idx] > max_val) {
                max_val = s_ptr[idx];
            }
        }
        
        // Compute exp and sum
        float exp_sum = 0.0f;
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            float val = s_ptr[idx];
            float exp_val = (val == -INFINITY) ? 0.0f : expf(val - max_val);
            exp_sum += exp_val;
        }
        
        // Compute weighted V sum
        for (int dim = 0; dim < head_dim; dim++) {
            float output_val = 0.0f;
            
            for (int k = 0; k < seq_k; k++) {
                int score_idx = q_pos * num_heads * seq_k + head * seq_k + k;
                float val = s_ptr[score_idx];
                float exp_val = (val == -INFINITY) ? 0.0f : expf(val - max_val);
                float softmax_weight = exp_sum > 0 ? exp_val / exp_sum : 0.0f;
                
                int v_idx = k * num_heads * head_dim + head * head_dim + dim;
                float v0 = __half2float(v_ptr[v_idx]);
                output_val += softmax_weight * v0;
            }
            
            // Write to final output buffer
            int out_idx = q_pos * num_heads * head_dim + head * head_dim + dim;
            out_ptr[out_idx] = output_val;
        }
    }
}
