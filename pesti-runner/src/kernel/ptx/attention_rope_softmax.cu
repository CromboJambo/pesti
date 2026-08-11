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
    // Shared memory for block-level reduction (one row per block)
    __shared__ float shared_scores[64];  // One score per thread in x-dimension
    
    // Get tile coordinates (each block handles one 64x64 tile)
    int tile_row = blockIdx.x;
    int tile_col = blockIdx.y;
    
    // Calculate sequence start positions for this tile
    int q_start = tile_row * 64;
    int k_start = tile_col * 64;
    
    // Get thread index within block
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
    // Calculate global positions for this thread
    int q_pos = q_start + tid_x;
    int k_pos = k_start + tid_y;
    
    // Bounds check
    if (q_pos >= seq_q || k_pos >= seq_k) return;
    
    // Get RoPE position (use sequence position as position embedding)
    int pos = q_pos;  // For causal attention, query position determines RoPE
    
    // Compute cos/sin for this position using standard RoPE formula
    float half_dim = head_dim / 2.0f;
    float inv_freq = 1.0f / powf(rope_base, (float)(tid_y % ((int)half_dim)) / half_dim);
    float freq = pos * inv_freq;
    float cos_val = cosf(freq);
    float sin_val = sinf(freq);
    
    // Load Q element (f16 -> f32) - first head, two dimensions at a time
    int q_idx = q_pos * num_heads * head_dim + tid_y * 2;
    float q0 = __half2float(q_ptr[q_idx]);
    float q1 = __half2float(q_ptr[q_idx + 1]);
    
    // Load K element (f16 -> f32) - first head, same dimensions
    int k_idx = k_pos * num_heads * head_dim + tid_y * 2;
    float k0 = __half2float(k_ptr[k_idx]);
    float k1 = __half2float(k_ptr[k_idx + 1]);
    
    // Apply RoPE rotation to Q pair
    apply_rope_pair(q0, q1, cos_val, sin_val);
    
    // Apply scaling factor (only to first dimension for simplicity)
    q0 *= scale;
    
    // Compute dot product of rotated Q with K (simplified: single pair)
    float score = q0 * k0 + q1 * k1;
    
    // Store in shared memory for reduction
    shared_scores[tid_x] = score;
    __syncthreads();
    
    // Apply causal mask: set to -inf where q_pos >= k_pos
    if (q_pos >= k_pos) {
        shared_scores[tid_x] = -1e9f;
    }
    __syncthreads();
    
    // Simple block-level max reduction for softmax numerically stable version
    float max_val = shared_scores[tid_x];
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (tid_x < stride) {
            max_val = fmaxf(max_val, shared_scores[tid_x + stride]);
        }
        __syncthreads();
    }
    
    // Compute exp and sum for softmax
    float exp_sum = 0.0f;
    float local_exp = expf(shared_scores[tid_x] - max_val);
    exp_sum += local_exp;
    shared_scores[tid_x] = local_exp;
    __syncthreads();
    
    // Parallel sum reduction
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (tid_x < stride) {
            exp_sum += shared_scores[tid_x + stride];
        }
        __syncthreads();
    }
    
    // Compute final softmax value
    float softmax_val = local_exp / exp_sum;
    
    // Store attention score (f32 output)
    s_ptr[q_pos * seq_k + k_pos] = softmax_val;
}
