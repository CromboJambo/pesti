// Optimized fused attention kernel with vectorized half2 loads
// and pre-computed RoPE cos/sin values

#include <cuda_fp16.h>
#include <math.h>

__device__ __forceinline__ void apply_rope_pair(
    float& q0, float& q1, 
    float cos_val, float sin_val
) {
    float new_q0 = q0 * cos_val - q1 * sin_val;
    float new_q1 = q0 * sin_val + q1 * cos_val;
    q0 = new_q0;
    q1 = new_q1;
}

__global__ void fused_attention_kernel_vectorized(
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
    // Each thread handles one (q_pos, k_pos, head) triplet
    int tid = threadIdx.x;
    int q_pos = blockIdx.x * blockDim.x + tid;
    int k_pos = blockIdx.y;
    int head = blockIdx.z;
    
    if (q_pos >= seq_q || k_pos >= seq_k || head >= num_heads) return;
    
    // Pre-compute RoPE cos/sin for this q_pos and dimension pair
    // We'll compute on-the-fly per dimension pair to avoid shared memory complexity
    
    float dot_product = 0.0f;
    
    // Vectorized loop: process 2 dimensions at a time using half2
    for (int d = 0; d < head_dim; d += 2) {
        // Compute RoPE position and cos/sin
        int pos = q_pos;
        float half_dim = head_dim / 2.0f;
        int dim_pair = d / 2;
        float inv_freq = 1.0f / powf(rope_base, (float)dim_pair / half_dim);
        float freq = pos * inv_freq;
        float cos_val = cosf(freq);
        float sin_val = sinf(freq);
        
        // Vectorized load: load 2 f16 values as half2
        // Q: [q0, q1] at position (q_pos, head) with offset d
        int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
        half2 q_pair = *(__half2*)&q_ptr[q_idx];
        
        // K: [k0, k1] at position (k_pos, head) with offset d  
        int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
        half2 k_pair = *(__half2*)&k_ptr[k_idx];
        
        // Convert to float for computation
        float2 q_f2 = __half22float2(q_pair);
        float2 k_f2 = __half22float2(k_pair);
        
        // Apply RoPE rotation
        apply_rope_pair(q_f2.x, q_f2.y, cos_val, sin_val);
        apply_rope_pair(k_f2.x, k_f2.y, cos_val, sin_val);
        
        // Accumulate dot product (vectorized: q0*k0 + q1*k1)
        dot_product += q_f2.x * k_f2.x + q_f2.y * k_f2.y;
    }
    
    // Scale by 1/sqrt(head_dim)
    dot_product *= scale;
    
    // Apply causal mask BEFORE softmax (q_pos >= k_pos = -inf)
    if (q_pos >= k_pos) {
        s_ptr[q_pos * seq_k + k_pos] = -1e9f;
        return;
    }
    
    // Store scaled attention score
    s_ptr[q_pos * seq_k + k_pos] = dot_product;
}

__global__ void fused_attention_kernel_softmax(
    float* __restrict__ s_ptr,
    int seq_q,
    int seq_k
) {
    int q_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_pos >= seq_q) return;
    
    // Find max for numerical stability
    float max_val = -1e30f;
    for (int k = 0; k < seq_k; k++) {
        float val = s_ptr[q_pos * seq_k + k];
        if (val > max_val) max_val = val;
    }
    
    // Compute exp and normalize
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
