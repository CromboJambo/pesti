// Flash Attention Kernel - Simplified Implementation (CUDA C++)
// Fused Q @ K^T + softmax + V computation
// Target: sm_89 (RTX 4070 Ti SUPER)
// Based on https://arxiv.org/abs/2205.14135

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

typedef half half_t;

__global__ void flash_attention_kernel(
    float scale,
    const half_t* __restrict__ q,      // [seq_q, num_heads, head_dim]
    const half_t* __restrict__ k,      // [seq_k, num_heads, head_dim]
    const half_t* __restrict__ v,      // [seq_k, num_heads, head_dim]
    float* __restrict__ o,             // [seq_q, num_heads, head_dim]
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim
) {
    // Shared memory for tiling (Q tile + K tile + O partial)
    extern __shared__ unsigned char smem[];
    
    float* q_tile = reinterpret_cast<float*>(smem);           // Offset 0
    float* k_tile = q_tile + head_dim * blockDim.x;          // Offset head_dim * threads
    float* o_tile = k_tile + head_dim * blockDim.x;          // Offset 2 * head_dim * threads
    
    // Thread/block indices
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_z = threadIdx.z;
    
    int grid_x = blockIdx.x;  // seq_q index
    int grid_y = blockIdx.y;  // head index
    
    // Global sequence indices
    int seq_q_idx = grid_x;
    int head_idx = grid_y;
    
    // Load parameters
    float q_val, k_val, v_val, o_val, qk_dot, max_val, exp_val;
    
    // Main loop over sequence positions (seq_k dimension)
    // Tiling: process seq_k in chunks of 128 tokens
    for (int seq_k_start = 0; seq_k_start < seq_k; seq_k_start += blockDim.x) {
        int seq_k_idx = seq_k_start + tid_x;
        
        // Initialize output partial sum (o = 0)
        o_val = 0.0f;
        
        // Load Q value for this position
        if (seq_q_idx < seq_q && head_idx < num_heads && tid_x < head_dim) {
            int q_offset = (seq_q_idx * num_heads + head_idx) * head_dim + tid_x;
            q_val = __half2float(q[q_offset]);
        } else {
            q_val = 0.0f;
        }
        
        // Load K value for this position
        if (seq_k_idx < seq_k && head_idx < num_heads && tid_x < head_dim) {
            int k_offset = (seq_k_idx * num_heads + head_idx) * head_dim + tid_x;
            k_val = __half2float(k[k_offset]);
        } else {
            k_val = 0.0f;
        }
        
        // Compute Q @ K^T (dot product of Q and K tiles)
        qk_dot = 0.0f;
        for (int dim = 0; dim < head_dim; dim++) {
            int q_offset = (seq_q_idx * num_heads + head_idx) * head_dim + dim;
            int k_offset = (seq_k_idx * num_heads + head_idx) * head_dim + dim;
            
            float q_tile_val = __half2float(q[q_offset]);
            float k_tile_val = __half2float(k[k_offset]);
            
            qk_dot += q_tile_val * k_tile_val;
        }
        
        // Apply scale (1/sqrt(head_dim))
        qk_dot *= scale;
        
        // Store softmax result in shared memory
        if (seq_k_idx < seq_k) {
            int smem_offset = tid_x + head_dim * tid_y;
            k_tile[smem_offset] = qk_dot;  // Store QK dot product
        }
        
        // Synchronize threads
        __syncthreads();
        
        // Find max for numerical stability (reduce across block)
        float local_max = -INFINITY;
        if (seq_k_idx < seq_k) {
            local_max = k_tile[tid_x];
        }
        
        // Parallel reduction to find global max
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid_x < offset) {
                float other_val = k_tile[tid_x + offset];
                local_max = fmaxf(local_max, other_val);
            }
            __syncthreads();
        }
        
        // Broadcast max to all threads
        float global_max = k_tile[tid_x];
        if (seq_k_idx >= seq_k) {
            global_max = -INFINITY;
        }
        
        // Compute exp(qk_dot - max_val)
        exp_val = expf(k_tile[tid_x] - global_max);
        
        // Store softmax result in shared memory
        o_tile[tid_x] = exp_val;  // Store exp(qk_dot - max_val)
        
        // Synchronize threads
        __syncthreads();
        
        // Compute sum of exp values (for normalization)
        float local_sum = 0.0f;
        if (seq_k_idx < seq_k) {
            local_sum = o_tile[tid_x];
        }
        
        // Parallel reduction to find global sum
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid_x < offset) {
                float other_val = o_tile[tid_x + offset];
                local_sum += other_val;
            }
            __syncthreads();
        }
        
        // Normalize softmax
        float softmax_val = exp_val / (local_sum + 1e-6f);
        
        // Load V value and compute S @ V
        if (seq_k_idx < seq_k && head_idx < num_heads) {
            int v_offset = (seq_k_idx * num_heads + head_idx) * head_dim + tid_x;
            float v_tile_val = __half2float(v[v_offset]);
            
            o_val += softmax_val * v_tile_val;
        }
        
        // Synchronize threads before next tile
        __syncthreads();
    }
    
    // Store final output
    if (seq_q_idx < seq_q && head_idx < num_heads && tid_x < head_dim) {
        int o_offset = (seq_q_idx * num_heads + head_idx) * head_dim + tid_x;
        o[o_offset] = o_val;
    }
}

// Kernel launcher
void launch_flash_attention(
    const half_t* d_q,
    const half_t* d_k,
    const half_t* d_v,
    float* d_o,
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim,
    cudaStream_t stream
) {
    // Configuration
    int block_size = 128;  // Threads per block
    int grid_x = seq_q;    // Grid dimensions
    int grid_y = num_heads;
    
    // Shared memory size (Q tile + K tile + O tile)
    int smem_size = 3 * head_dim * block_size * sizeof(float);
    
    // Launch kernel
    flash_attention_kernel<<<grid_x, grid_y, smem_size, stream>>>(
        1.0f / sqrtf(head_dim),
        d_q,
        d_k,
        d_v,
        d_o,
        seq_q,
        seq_k,
        num_heads,
        head_dim
    );
}
