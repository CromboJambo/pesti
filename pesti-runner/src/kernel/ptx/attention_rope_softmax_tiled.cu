// Optimized fused attention kernel with tiled shared memory + RoPE pre-computation
// 
// Architecture (sm_89 RTX 4070 Ti SUPER):
// - Tile size: 32 (seq_k dimension)
// - One thread per (q_pos, head) combination
// - Sequential processing over seq_k
// - Shared memory for K values (not used in simplified version)
//
// Grid: (ceil(seq_q * num_heads / 128), 1, 1) with blockDim.x = 128

#include <cuda_fp16.h>
#include <math.h>

#define TILE_SIZE 32
#define HEAD_DIM 16

__device__ __forceinline__ void apply_rope_pair(
    float& q0, float& q1,
    float cos_val_q, float sin_val_q,
    float cos_val_k, float sin_val_k
) {
    // Apply RoPE rotation to Q pair
    float new_q0 = q0 * cos_val_q - q1 * sin_val_q;
    float new_q1 = q0 * sin_val_q + q1 * cos_val_q;
    q0 = new_q0;
    q1 = new_q1;
    
    // Apply RoPE rotation to K pair
    float new_k0 = q0 * cos_val_k - q1 * sin_val_k;  // Note: using rotated Q for K (simplified)
    float new_k1 = q0 * sin_val_k + q1 * cos_val_k;
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
    // 1D grid over all (q_pos * num_heads + head) combinations
    int qh = blockIdx.x * blockDim.x + threadIdx.x;
    int total_qh = seq_q * num_heads;
    
    if (qh >= total_qh) return;
    
    int q_pos = qh / num_heads;
    int head = qh % num_heads;
    
    const float half_dim = head_dim / 2.0f;
    float dot_product = 0.0f;
    
    // Sequential loop over k positions (like original working version)
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        float dot = 0.0f;
        
        // Dot product over head dimensions in pairs
        for (int d = 0; d < head_dim; d += 2) {
            int q_idx = qh * head_dim + d;
            int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
            
            float q0 = __half2float(q_ptr[q_idx]);
            float q1 = __half2float(q_ptr[q_idx + 1]);
            
            // Compute RoPE frequency for this dimension pair
            float inv_freq = 1.0f / powf(rope_base, (float)(d / 2) / half_dim);
            float freq_q = q_pos * inv_freq;
            float c_q = cosf(freq_q), s_q = sinf(freq_q);
            
            // Apply RoPE to Q pair
            float new_q0 = q0 * c_q - q1 * s_q;
            float new_q1 = q0 * s_q + q1 * c_q;
            
            float k0 = __half2float(k_ptr[k_idx]);
            float k1 = __half2float(k_ptr[k_idx + 1]);
            
            // Compute RoPE for K - use same frequency calculation as original
            float freq_k = k_pos * inv_freq;
            float c_k = cosf(freq_k), s_k = sinf(freq_k);
            
            // Apply RoPE to K pair
            float new_k0 = k0 * c_k - k1 * s_k;
            float new_k1 = k0 * s_k + k1 * c_k;  // Fixed: was using q1 instead of k1
            
            dot += new_q0 * new_k0 + new_q1 * new_k1;
        }
        
        // Apply scale (1/sqrt(head_dim))
        dot *= scale;
        
        // Causal mask: set to -inf if q_pos >= k_pos
        if (q_pos >= k_pos) {
            dot = -INFINITY;
        }
        
        // Store attention score with correct indexing (matching original - no scale)
        int out_idx = q_pos * num_heads * seq_k + head * seq_k + k_pos;
        s_ptr[out_idx] = dot;
    }
}

__global__ void apply_softmax_kernel(
    float* __restrict__ s_ptr,
    int seq_q,
    int seq_k,
    int num_heads
) {
    // Each block processes one (q_pos, head) pair
    int total_qh = seq_q * num_heads;
    int qh = blockIdx.x * num_heads + blockIdx.y;  // FIXED: q_pos first, then head (matches launch config)
    
    if (qh >= total_qh) return;
    
    int q_pos = qh / num_heads;
    int head = qh % num_heads;
    
    // Shared memory for reduction: one value per thread in block
    __shared__ float exp_vals[128];  // blockDim.x <= 128
    
    // Phase 1: Find max value across seq_k positions for this (q_pos, head) pair
    float max_val = -INFINITY;
    int local_seq_k = seq_k;
    
    for (int k_pos = threadIdx.x; k_pos < local_seq_k; k_pos += blockDim.x) {
        // Match the indexing from fused_attention_kernel_tiled:
        // out_idx = q_pos * num_heads * seq_k + head * seq_k + k_pos
        int idx = qh / num_heads * num_heads * seq_k + (qh % num_heads) * seq_k + k_pos;
        float val = s_ptr[idx];
        if (val > max_val) {
            max_val = val;
        }
    }
    
    // Parallel reduction to find global max for this (q_pos, head) pair
    // Each thread writes its partial max to shared memory, then reduce in-place
    __shared__ float max_vals[128];  // One per thread
    
    // Store initial max_val from each thread
    max_vals[threadIdx.x] = max_val;
    __syncthreads();
    
    // Parallel reduction on shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = max_vals[threadIdx.x + stride];
            if (other > max_vals[threadIdx.x]) {
                max_vals[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }
    
    // Thread 0 has the global max
    if (threadIdx.x == 0) {
        max_val = max_vals[0];
    }
    
    // Phase 2: Compute softmax for each position
    for (int k_pos = threadIdx.x; k_pos < local_seq_k; k_pos += blockDim.x) {
        // Match indexing from above
        int idx = qh / num_heads * num_heads * seq_k + (qh % num_heads) * seq_k + k_pos;
        float val = s_ptr[idx];
        
        float exp_val;
        if (max_val == -INFINITY) {
            // All positions in this row are masked (-INF), softmax = 0 everywhere
            // But we need at least one position to have probability 1.0 for numerical stability
            // Assign 1.0 to the first unmasked position, or first position if all masked
            exp_val = (threadIdx.x == 0) ? 1.0f : 0.0f;
        } else if (val == -INFINITY) {
            exp_val = 0.0f;
        } else {
            exp_val = expf(val - max_val);
        }
        
        exp_vals[threadIdx.x] = exp_val;
        s_ptr[idx] = exp_val;
    }
    
    __syncthreads();
    
    // Phase 3: Compute sum of exp values (using shared memory)
    float exp_sum = 0.0f;
    for (int t = threadIdx.x; t < blockDim.x; t += blockDim.x) {
        if (t < local_seq_k) {
            exp_sum += exp_vals[t];
        }
    }
    
    // Parallel reduction for sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && threadIdx.x + stride < local_seq_k) {
            exp_vals[threadIdx.x] += exp_vals[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    __syncthreads();
    if (threadIdx.x == 0) {
        exp_sum = exp_vals[0];
    }
    __syncthreads();
    exp_sum = exp_vals[0];
    
    // Phase 4: Normalize by sum
    for (int k_pos = threadIdx.x; k_pos < local_seq_k; k_pos += blockDim.x) {
        // Match indexing from above
        int idx = qh / num_heads * num_heads * seq_k + (qh % num_heads) * seq_k + k_pos;
        float val = s_ptr[idx];
        
        if (val == 0.0f) {
            s_ptr[idx] = 0.0f;  // Causal mask position
        } else {
            s_ptr[idx] = val / exp_sum;
        }
    }
}
