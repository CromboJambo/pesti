//! Fused RoPE + Attention + Softmax + V-Multiplication kernel
// Uses shared memory for exp_sum to avoid score buffer corruption
// OPTIMIZED: Parallelized softmax computation across all threads in kernel 2

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
    
    // Half-swap RoPE rotation (matches llama.cpp / HuggingFace transformers)
    for (int chunk = threadIdx.x; chunk < head_dim / 2; chunk += blockDim.x) {
        int d = chunk;
        
        int q_idx_first = q_pos * num_heads * head_dim + head * head_dim + d;
        int q_idx_second = q_pos * num_heads * head_dim + head * head_dim + (d + head_dim / 2);
        
        int k_idx_first = k_pos * num_heads * head_dim + head * head_dim + d;
        int k_idx_second = k_pos * num_heads * head_dim + head * head_dim + (d + head_dim / 2);
        
        float q_first = __half2float(q_ptr[q_idx_first]);
        float q_second = __half2float(q_ptr[q_idx_second]);
        
        float k_first = __half2float(k_ptr[k_idx_first]);
        float k_second = __half2float(k_ptr[k_idx_second]);
        
        // Apply RoPE to Q (rotated by q_pos) before dot product
        float inv_freq_q = 1.0f / powf(rope_base, (float)d / ((float)head_dim / 2.0f));
        float freq_q = (float)q_pos * inv_freq_q;
        float cos_val_q = cosf(freq_q);
        float sin_val_q = sinf(freq_q);
        
        float q_first_rope = q_first * cos_val_q - q_second * sin_val_q;
        float q_second_rope = q_first * sin_val_q + q_second * cos_val_q;
        
        // Apply RoPE to K (rotated by k_pos) before dot product
        float inv_freq_k = 1.0f / powf(rope_base, (float)d / ((float)head_dim / 2.0f));
        float freq_k = (float)k_pos * inv_freq_k;
        float cos_val_k = cosf(freq_k);
        float sin_val_k = sinf(freq_k);
        
        float k_first_rope = k_first * cos_val_k - k_second * sin_val_k;
        float k_second_rope = k_first * sin_val_k + k_second * cos_val_k;
        
        dot_product += q_first_rope * k_first_rope + q_second_rope * k_second_rope;
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
// OPTIMIZED: All threads cooperate on each phase instead of tid==0 only
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
    
    int score_offset = q_pos * num_heads * seq_k + head * seq_k;
    
    // Pass 1: Parallel max-finding across all threads
    float local_max = -INFINITY;
    for (int k = tid; k < seq_k; k += blockDim.x) {
        int idx = score_offset + k;
        if (s_ptr[idx] > local_max) {
            local_max = s_ptr[idx];
        }
    }
    
    // Reduce max across threads using shared memory
    __shared__ float thread_maxes[32];
    thread_maxes[tid] = local_max;
    __syncthreads();
    
    float global_max = -INFINITY;
    for (int t = 0; t < blockDim.x; t++) {
        if (thread_maxes[t] > global_max) {
            global_max = thread_maxes[t];
        }
    }
    
    // Pass 2: Parallel exp computation with subtraction of max
    float local_sum = 0.0f;
    for (int k = tid; k < seq_k; k += blockDim.x) {
        int idx = score_offset + k;
        float val = s_ptr[idx];
        float exp_val = (val == -INFINITY) ? 0.0f : expf(val - global_max);
        s_ptr[idx] = exp_val;
        local_sum += exp_val;
    }
    
    // Reduce sum across threads using shared memory
    __shared__ float thread_sums[32];
    thread_sums[tid] = local_sum;
    __syncthreads();
    
    float total_sum = 0.0f;
    for (int t = 0; t < blockDim.x; t++) {
        total_sum += thread_sums[t];
    }
    
    // Store exp_sum in shared memory for normalization
    if (tid == 0) {
        shared_exp_sum[0] = total_sum;
    }
    __syncthreads();
    
    float exp_sum = shared_exp_sum[0];
    
    // Pass 3: Parallel normalization of softmax weights
    if (exp_sum > 0) {
        for (int k = tid; k < seq_k; k += blockDim.x) {
            int idx = score_offset + k;
            s_ptr[idx] /= exp_sum;
        }
    }
    
    __syncthreads();
    
    // Pass 4: Parallel weighted sum of V for each output dimension
    // Each thread handles different head_dim positions (grid-stride loop)
    for (int dim_idx = tid; dim_idx < head_dim; dim_idx += blockDim.x) {
        float output_val = 0.0f;
        
        for (int k = 0; k < seq_k; k++) {
            int score_idx = score_offset + k;
            float softmax_val = s_ptr[score_idx];
            
            int v_idx = k * num_heads * head_dim + head * head_dim + dim_idx;
            float v0 = __half2float(v_ptr[v_idx]);
            output_val += softmax_val * v0;
        }
        
        // Write output to new layout [seq_q, num_heads, head_dim] at end of buffer
        int score_buffer_size = seq_q * num_heads * seq_k;
        int out_idx = score_buffer_size + q_pos * num_heads * head_dim + head * head_dim + dim_idx;
        s_ptr[out_idx] = output_val;
    }
}
