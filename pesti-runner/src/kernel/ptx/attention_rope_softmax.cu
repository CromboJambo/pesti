//! Fused RoPE + Attention + Softmax + V-Multiplication kernel
// Uses shared memory for exp_sum to avoid score buffer corruption

#include <cuda_fp16.h>
#include <math.h>

// Kernel 1: Compute raw attention scores with RoPE and causal mask
__global__ void fused_attention_kernel(
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
    int q_pos = blockIdx.x;
    int k_pos = blockIdx.y;
    int head = blockIdx.z;
    
    if (q_pos >= seq_q || k_pos >= seq_k) return;
    
    extern __shared__ float shared_dot[];
    
    float dot_product = 0.0f;
    
    for (int chunk = threadIdx.x; chunk < head_dim / 2; chunk += blockDim.x) {
        int d = chunk * 2;
        
        int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
        float q0 = __half2float(q_ptr[q_idx]);
        float q1 = __half2float(q_ptr[q_idx + 1]);
        
        int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
        float k0 = __half2float(k_ptr[k_idx]);
        float k1 = __half2float(k_ptr[k_idx + 1]);
        
        // Apply RoPE to Q (rotated by q_pos) and K (rotated by k_pos) before dot product
        float inv_freq_q = 1.0f / powf(rope_base, (float)d / ((float)head_dim / 2.0f));
        float freq_q = (float)q_pos * inv_freq_q;
        float cos_val_q = cosf(freq_q);
        float sin_val_q = sinf(freq_q);
        
        // RoPE on Q
        float q0_rope = q0 * cos_val_q - q1 * sin_val_q;
        float q1_rope = q0 * sin_val_q + q1 * cos_val_q;
        
        // Apply RoPE to K (rotated by k_pos)
        float inv_freq_k = 1.0f / powf(rope_base, (float)d / ((float)head_dim / 2.0f));
        float freq_k = (float)k_pos * inv_freq_k;
        float cos_val_k = cosf(freq_k);
        float sin_val_k = sinf(freq_k);
        
        // RoPE on K
        float k0_rope = k0 * cos_val_k - k1 * sin_val_k;
        float k1_rope = k0 * sin_val_k + k1 * cos_val_k;
        
        dot_product += q0_rope * k0_rope + q1_rope * k1_rope;
    }
    
    shared_dot[threadIdx.x] = dot_product;
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int t = 0; t < blockDim.x; t++) {
            total += shared_dot[t];
        }
        
        total *= scale;
        
        if (k_pos > q_pos) {
            total = -INFINITY;
        }
        
        int out_idx = q_pos * num_heads * seq_k + head * seq_k + k_pos;
        s_ptr[out_idx] = total;
    }
}

// Kernel 2: Apply softmax AND multiply by V to get final output
__global__ void apply_softmax_and_output_kernel(
    float* __restrict__ s_ptr,      // IN/OUT: scores → output
    const half* __restrict__ v_ptr, // values: [seq_k, num_heads, head_dim]
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim
) {
    extern __shared__ float shared_exp_sum[];  // Shared memory for exp_sum
    
    int q_pos = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    
    if (q_pos >= seq_q || head >= num_heads) return;
    
    // Pass 1: Find max and compute exp values for this (q_pos, head) pair
    if (tid == 0) {
        float max_val = -INFINITY;
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            if (s_ptr[idx] > max_val) {
                max_val = s_ptr[idx];
            }
        }
        
        float exp_sum = 0.0f;
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            float val = s_ptr[idx];
            float exp_val = (val == -INFINITY) ? 0.0f : expf(val - max_val);
            s_ptr[idx] = exp_val;
            exp_sum += exp_val;
        }
        
        // Store exp_sum in shared memory instead of score buffer!
        shared_exp_sum[0] = exp_sum;
    }
    
    __syncthreads();
    
    float exp_sum = shared_exp_sum[0];
    
    // Pass 2: Normalize softmax weights (all scores, no corruption!)
    if (tid == 0 && exp_sum > 0) {
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            s_ptr[idx] /= exp_sum;
        }
    }
    
    __syncthreads();
    
    // Pass 3: Compute weighted sum of V for each output dimension
    int dim_idx = tid;
    
    while (dim_idx < head_dim) {
        float output_val = 0.0f;
        
        for (int k = 0; k < seq_k; k++) {
            int score_idx = q_pos * num_heads * seq_k + head * seq_k + k;
            float softmax_val = s_ptr[score_idx];  // Read normalized weight
            
            int v_idx = k * num_heads * head_dim + head * head_dim + dim_idx;
            float v0 = __half2float(v_ptr[v_idx]);
            output_val += softmax_val * v0;
        }
        
        // Write output to new layout [seq_q, num_heads, head_dim] at end of buffer
        int score_buffer_size = seq_q * num_heads * seq_k;  // In elements (floats)
        int out_idx = score_buffer_size + q_pos * num_heads * head_dim + head * head_dim + dim_idx;
        s_ptr[out_idx] = output_val;
        
        dim_idx += blockDim.x;
    }
}
