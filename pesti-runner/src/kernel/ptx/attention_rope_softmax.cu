#include <cuda_fp16.h>
#include <math.h>

// Apply RoPE rotation to a pair of dimensions
__device__ __forceinline__ void apply_rope_pair(
    float& q0, float& q1, 
    float cos_val, float sin_val
) {
    // RoPE rotation: [q0, q1] -> [q0*cos - q1*sin, q0*sin + q1*cos]
    float new_q0 = q0 * cos_val - q1 * sin_val;
    float new_q1 = q0 * sin_val + q1 * cos_val;
    q0 = new_q0;
    q1 = new_q1;
}

__global__ void fused_attention_kernel(
    float scale,              // 1/sqrt(head_dim)
    const half* __restrict__ q_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    float* __restrict__ s_ptr,
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim,
    float rope_base,
    int max_pos
) {
    // Each thread handles one (q_pos, k_pos) pair
    // Block is 1D with 128 threads
    int tid = threadIdx.x;
    int q_pos = blockIdx.x * blockDim.x + tid;
    
    // Only threads within bounds participate
    if (q_pos >= seq_q) return;
    
    // Compute dot product across ALL dimensions for this (q_pos, k_pos) pair
    // We iterate over k_pos sequentially (one thread per q_pos)
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        // Causal mask: skip if q_pos >= k_pos
        if (q_pos >= k_pos) {
            s_ptr[q_pos * seq_k + k_pos] = -1e9f;
            continue;
        }
        
        float dot_product = 0.0f;
        
        // Iterate over all head dimensions
        for (int d = 0; d < head_dim; d += 2) {
            // Compute RoPE position
            int pos = q_pos;
            float half_dim = head_dim / 2.0f;
            int dim_pair = d / 2;
            float inv_freq = 1.0f / powf(rope_base, (float)dim_pair / half_dim);
            float freq = pos * inv_freq;
            float cos_val = cosf(freq);
            float sin_val = sinf(freq);
            
            // Load Q elements for this head and dimension pair
            int q_idx = q_pos * num_heads * head_dim + d;
            float q0 = __half2float(q_ptr[q_idx]);
            float q1 = __half2float(q_ptr[q_idx + 1]);
            
            // Apply RoPE rotation to Q
            apply_rope_pair(q0, q1, cos_val, sin_val);
            
            // Load K elements for same dimensions
            int k_idx = k_pos * num_heads * head_dim + d;
            float k0 = __half2float(k_ptr[k_idx]);
            float k1 = __half2float(k_ptr[k_idx + 1]);
            
            // Apply RoPE rotation to K (same position as Q for this pair)
            apply_rope_pair(k0, k1, cos_val, sin_val);
            
            // Accumulate dot product
            dot_product += q0 * k0 + q1 * k1;
        }
        
        // Scale by 1/sqrt(head_dim)
        dot_product *= scale;
        
        // Store scaled attention score
        s_ptr[q_pos * seq_k + k_pos] = dot_product;
    }
    
    // Now apply softmax per query row
    // Find max for numerical stability
    float max_val = -1e30f;
    for (int k = 0; k < seq_k; k++) {
        float val = s_ptr[q_pos * seq_k + k];
        if (val > max_val) max_val = val;
    }
    
    // Compute exp and sum
    float exp_sum = 0.0f;
    for (int k = 0; k < seq_k; k++) {
        float val = s_ptr[q_pos * seq_k + k];
        float exp_val = expf(val - max_val);
        s_ptr[q_pos * seq_k + k] = exp_val;
        exp_sum += exp_val;
    }
    
    // Normalize
    for (int k = 0; k < seq_k; k++) {
        s_ptr[q_pos * seq_k + k] /= exp_sum;
    }
}
