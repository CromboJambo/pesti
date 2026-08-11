// Optimized fused attention kernel with vectorized half2 loads
// 
// Architecture:
// - One thread per (head, q_pos) pair
// - Vectorized memory loads for better bandwidth
// - Simple sequential processing (no shared memory complexity yet)

#include <cuda_fp16.h>
#include <math.h>

#define HEAD_DIM 16     // Fixed head dimension for this kernel

__device__ __forceinline__ void apply_rope_pair(
    float& q0, float& q1, 
    float cos_val, float sin_val
) {
    float new_q0 = q0 * cos_val - q1 * sin_val;
    float new_q1 = q0 * sin_val + q1 * cos_val;
    q0 = new_q0;
    q1 = new_q1;
}

__device__ __forceinline__ void apply_rope_pair_k(
    float& k0, float& k1, 
    float cos_val, float sin_val
) {
    float new_k0 = k0 * cos_val - k1 * sin_val;
    float new_k1 = k0 * sin_val + k1 * cos_val;
    k0 = new_k0;
    k1 = new_k1;
}

__global__ void fused_attention_kernel_tiled(
    float scale,
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
    // One thread per (head, q_pos) pair - simple sequential processing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= seq_q * num_heads) return;
    
    int head = idx / seq_q;
    int q_pos = idx % seq_q;
    
    float half_dim = head_dim / 2.0f;
    float dot_product = 0.0f;
    
    // Sequential loop over all k positions (like original working version)
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        // Load K tile (vectorized half2) - process 2 dimensions per load
        for (int d = 0; d < head_dim; d += 2) {
            int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
            half2 k_pair = *(__half2*)&k_ptr[k_idx];
            
            // Load Q (vectorized half2)
            int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
            half2 q_pair = *(__half2*)&q_ptr[q_idx];
            
            float2 k_f2 = __half22float2(k_pair);
            float2 q_f2 = __half22float2(q_pair);
            
            // Apply RoPE to Q and K before dot product
            int dim_pair = d / 2;
            float inv_freq = 1.0f / powf(rope_base, (float)dim_pair / half_dim);
            float freq_q = q_pos * inv_freq;
            float freq_k = k_pos * inv_freq;
            
            float cos_q = cosf(freq_q), sin_q = sinf(freq_q);
            float cos_k = cosf(freq_k), sin_k = sinf(freq_k);
            
            apply_rope_pair(q_f2.x, q_f2.y, cos_q, sin_q);
            apply_rope_pair_k(k_f2.x, k_f2.y, cos_k, sin_k);
            
            dot_product += q_f2.x * k_f2.x + q_f2.y * k_f2.y;
        }
        
        dot_product *= scale;
    }
    
    // Store attention scores (no softmax yet - for debugging)
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        int s_idx = q_pos * seq_k + k_pos;
        // Need to recompute dot_product with scale applied per k_pos
        float dot_product_k = 0.0f;
        for (int d = 0; d < head_dim; d += 2) {
            int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
            half2 k_pair = *(__half2*)&k_ptr[k_idx];
            
            int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
            half2 q_pair = *(__half2*)&q_ptr[q_idx];
            
            float2 k_f2 = __half22float2(k_pair);
            float2 q_f2 = __half22float2(q_pair);
            
            int dim_pair = d / 2;
            float inv_freq = 1.0f / powf(rope_base, (float)dim_pair / half_dim);
            float freq_q = q_pos * inv_freq;
            float freq_k = k_pos * inv_freq;
            
            float cos_q = cosf(freq_q), sin_q = sinf(freq_q);
            float cos_k = cosf(freq_k), sin_k = sinf(freq_k);
            
            apply_rope_pair(q_f2.x, q_f2.y, cos_q, sin_q);
            apply_rope_pair_k(k_f2.x, k_f2.y, cos_k, sin_k);
            
            dot_product_k += q_f2.x * k_f2.x + q_f2.y * k_f2.y;
        }
        s_ptr[s_idx] = dot_product_k * scale;
    }
}
