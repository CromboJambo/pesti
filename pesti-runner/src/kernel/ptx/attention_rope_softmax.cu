//! Minimal test with shared memory accumulation for correct dot product

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
    // Each block handles one (q_pos, k_pos, head) triplet
    int q_pos = blockIdx.x;
    int k_pos = blockIdx.y;
    int head = blockIdx.z;
    
    // Thread within block processes one pair of dimensions
    int tid = threadIdx.x;
    int half_dim = head_dim / 2;
    
    if (q_pos >= seq_q || k_pos >= seq_k) return;
    
    // Shared memory for partial dot products from each thread
    extern __shared__ float shared_dot[];
    
    float dot_product = 0.0f;
    
    for (int chunk = tid; chunk < half_dim; chunk += blockDim.x) {
        int d = chunk * 2;
        
        // Q layout: [seq_q, num_heads, head_dim]
        int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
        float q0 = __half2float(q_ptr[q_idx]);
        float q1 = __half2float(q_ptr[q_idx + 1]);
        
        // K layout: [seq_k, num_heads, head_dim]
        int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
        float k0 = __half2float(k_ptr[k_idx]);
        float k1 = __half2float(k_ptr[k_idx + 1]);
        
        // Accumulate dot product (raw, no RoPE for debugging)
        dot_product += q0 * k0 + q1 * k1;
    }
    
    // Store partial result in shared memory
    shared_dot[tid] = dot_product;
    
    // Synchronize so all threads have written their partial results
    __syncthreads();
    
    // Thread 0 sums up all partial results
    if (tid == 0) {
        float total = 0.0f;
        for (int t = 0; t < blockDim.x; t++) {
            total += shared_dot[t];
        }
        
        // Apply scale (1/sqrt(head_dim))
        total *= scale;
        
        // Causal mask: set to -inf if k_pos > q_pos (future tokens)
        if (k_pos > q_pos) {
            total = -INFINITY;
        }
        
        // Store raw score
        int out_idx = q_pos * num_heads * seq_k + head * seq_k + k_pos;
        s_ptr[out_idx] = total;
    }
}

// Kernel 2: Apply softmax per (q_pos, head) pair - thread 0 does all work
__global__ void apply_softmax_kernel(
    float* s_ptr,
    int seq_q,
    int seq_k,
    int num_heads
) {
    int q_pos = blockIdx.x;
    int head = blockIdx.y;

    if (q_pos >= seq_q || head >= num_heads) return;

    // Thread 0 does all softmax computation for this (q_pos, head) pair
    if (threadIdx.x == 0) {
        // Find max
        float max_val = -INFINITY;
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
            s_ptr[idx] = exp_val;  // Store exp value
            exp_sum += exp_val;
        }

        // Normalize
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            s_ptr[idx] /= exp_sum;
        }
    }
}
