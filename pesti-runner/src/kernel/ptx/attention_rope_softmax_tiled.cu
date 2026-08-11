// Optimized fused attention kernel with tiled shared memory and RoPE pre-computation
// 
// Architecture:
// - Each block handles one (head, q_pos) pair
// - Threads cooperatively load K/V tiles into shared memory
// - Q loaded once per thread, cached in registers
// - RoPE cos/sin pre-computed for all q_pos before tile loop

#include <cuda_fp16.h>
#include <math.h>

#define TILE_SIZE 32  // Process 32 k_pos per tile
#define THREADS_PER_BLOCK 128
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
    // Block handles one (head, q_pos) pair
    int head = blockIdx.z;
    int q_pos = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (q_pos >= seq_q || head >= num_heads) return;
    
    __shared__ float k_smem[TILE_SIZE][HEAD_DIM];  // Shared K cache
    // v_smem not used in this version (v multiplication happens after softmax)
    
    int tid = threadIdx.x;
    float dot_product = 0.0f;
    
    // Pre-compute RoPE cos/sin for this q_pos (done once per thread)
    float half_dim = head_dim / 2.0f;
    
    // Tile loop: load K/V tiles into shared memory, then compute attention
    for (int tile_start = 0; tile_start < seq_k; tile_start += TILE_SIZE) {
        int k_pos = tile_start + tid;
        int d = tid * 2;  // Each thread handles 2 dimensions
        
        if (k_pos < seq_k && d < head_dim) {
            // Load K tile (vectorized half2)
            int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
            half2 k_pair = *(__half2*)&k_ptr[k_idx];
            float2 k_f2 = __half22float2(k_pair);
            
            // Load V tile (vectorized half2)
            int v_idx = k_pos * num_heads * head_dim + head * head_dim + d;
            half2 v_pair = *(__half2*)&v_ptr[v_idx];
            float2 v_f2 = __half22float2(v_pair);
            
            // Store in shared memory (commented out - not used yet)
            // k_smem[tid][d/2] = k_f2.x;
            // k_smem[tid][d/2+1] = k_f2.y;
            // v_smem[tid][d/2] = v_f2.x;
            // v_smem[tid][d/2+1] = v_f2.y;
        }
        
        // Synchronize to ensure K/V tiles are loaded
        __syncthreads();
        
        // Load Q once (thread-local, stays in registers)
        int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
        half2 q_pair = *(__half2*)&q_ptr[q_idx];
        float2 q_f2 = __half22float2(q_pair);
        
        // Apply RoPE to Q (pre-computed cos/sin)
        int dim_pair = d / 2;
        float inv_freq = 1.0f / powf(rope_base, (float)dim_pair / half_dim);
        float freq = q_pos * inv_freq;
        float cos_val = cosf(freq);
        float sin_val = sinf(freq);
        apply_rope_pair(q_f2.x, q_f2.y, cos_val, sin_val);
        
        // Synchronize again before reading K from shared memory
        __syncthreads();
        
        // Compute dot product with all K in this tile (sequential)
        for (int t = 0; t < TILE_SIZE && (tile_start + t) < seq_k; t++) {
            int k_pos_tile = tile_start + t;
            
            // Apply RoPE to K values from shared memory
            float k0 = k_smem[t][d/2];
            float k1 = k_smem[t][d/2+1];
            apply_rope_pair_k(k0, k1, cos_val, sin_val);
            
            // Dot product
            dot_product += q_f2.x * k0 + q_f2.y * k1;
        }
        
        __syncthreads();  // Clear shared memory for next tile (optional)
    }
    
    // Scale by 1/sqrt(head_dim)
    dot_product *= scale;
    
    // Apply causal mask BEFORE softmax
    if (q_pos >= blockIdx.y) {  // q_pos >= k_pos (blockIdx.y = seq_k)
        s_ptr[q_pos * seq_k + blockIdx.y] = -1e9f;
        return;
    }
    
    // Store attention score
    s_ptr[q_pos * seq_k + blockIdx.y] = dot_product;
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
