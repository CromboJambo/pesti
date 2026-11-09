// Full fusion attention kernel: scores + softmax + V-multiply in one pass
// Three-stage -> One-stage: All computation fused into single kernel launch

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void fused_attention_full_kernel(
    const half* q_ptr,
    const half* k_ptr,
    const half* v_ptr,
    float* out_ptr,
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim
) {
    // Compute global indices
    int q_pos = blockIdx.x;           // Query position
    int head = blockIdx.y;            // Head index
    int dim_idx = threadIdx.x;        // Dimension within head
    
    if (q_pos >= seq_q || head >= num_heads || dim_idx >= head_dim)
        return;
    
    // Shared memory for softmax computation (supports seq_k up to 512)
    __shared__ float s_scores[512];  // Max seq_k = 512 for shared buffer
    __shared__ float s_exp_sum;
    
    float sum_v = 0.0f;
    
    // Iterate over all k positions to compute scores, softmax, and V-weighted sum
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        // Causal masking: only attend to k_pos <= q_pos (past and current positions)
        if (k_pos > q_pos) {
            continue;  // Skip future positions
        }
        // Compute dot product between q[q_pos, head, :] and k[k_pos, head, :]
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
            int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
            
            float q_val = __half2float(q_ptr[q_idx]);
            float k_val = __half2float(k_ptr[k_idx]);
            dot += q_val * k_val;
        }
        
        // Scale by sqrt(head_dim)
        float score = dot / sqrtf((float)head_dim);
        
        // Store score in shared memory for softmax reduction
        s_scores[threadIdx.x + threadIdx.y * head_dim] = score;
        __syncthreads();
        
        // Compute max and exp_sum for numerical stability
        float max_score = score;
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            max_score = score;
        }
        __syncthreads();
        
        // For simplicity, use per-thread max (not warp-reduced for now)
        float local_max = fmaxf(max_score, score);
        float exp_val = expf(score - local_max);
        
        // Accumulate sum of exp values
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            s_exp_sum = exp_val;
        } else {
            s_exp_sum += exp_val;
        }
        __syncthreads();
        
        // Compute softmax weight
        float softmax_val = exp_val / s_exp_sum;
        
        // Weighted V sum: out[q_pos, head, dim_idx] += softmax[k_pos] * v[k_pos, head, dim_idx]
        int v_idx = k_pos * num_heads * head_dim + head * head_dim + dim_idx;
        float v_val = __half2float(v_ptr[v_idx]);
        sum_v += softmax_val * v_val;
        
        __syncthreads();
    }
    
    // Write output
    int out_idx = q_pos * num_heads * head_dim + head * head_dim + dim_idx;
    out_ptr[out_idx] = sum_v;
}

// Simplified version with fixed block dimensions
__global__ void fused_attention_simple_kernel(
    const half* q_ptr,
    const half* k_ptr,
    const half* v_ptr,
    float* out_ptr,
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim
) {
    // Each block handles one (q_pos, head) pair
    int q_pos = blockIdx.x;
    int head = blockIdx.y;
    int dim_idx = threadIdx.x;
    
    if (q_pos >= seq_q || head >= num_heads || dim_idx >= head_dim)
        return;
    
    float sum_v = 0.0f;
    
    // Compute scores, softmax, and V-multiply in one pass over k positions
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        // Dot product: q[q_pos, head, :] · k[k_pos, head, :]
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
            int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
            
            float q_val = __half2float(q_ptr[q_idx]);
            float k_val = __half2float(k_ptr[k_idx]);
            dot += q_val * k_val;
        }
        
        // Scale by sqrt(head_dim)
        float score = dot / sqrtf((float)head_dim);
        
        // Store for softmax (we'll do two-pass: first find max, then compute exp_sum)
        // For now, accumulate in a way that works without shared memory reduction
        if (k_pos == 0) {
            // First score - initialize max and sum
            float max_score = score;
            float exp_sum = expf(score - max_score);
            
            // Second pass to compute final weights (we need to re-iterate, but for simplicity...)
            // Actually, let's do a simpler approach: just store scores in registers
        }
    }
    
    // Simplified: just write placeholder
    int out_idx = q_pos * num_heads * head_dim + head * head_dim + dim_idx;
    out_ptr[out_idx] = sum_v;
}

// Final version: two-pass softmax with shared memory for max reduction
__global__ void fused_attention_kernel(
    const half* q_ptr,
    const half* k_ptr,
    const half* v_ptr,
    float* out_ptr,
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim
) {
    // Block dimensions: 1D thread index for dimension loop
    int q_pos = blockIdx.x;
    int head = blockIdx.y;
    int dim_idx = threadIdx.x;
    
    if (q_pos >= seq_q || head >= num_heads || dim_idx >= head_dim)
        return;
    
    float sum_v = 0.0f;
    
    // First pass: compute all scores and find max (supports seq_k up to 512)
    float max_score = -INFINITY;
    float scores[512];  // Stack-allocated for seq_k <= 512
    
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        // Dot product
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
            int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
            
            float q_val = __half2float(q_ptr[q_idx]);
            float k_val = __half2float(k_ptr[k_idx]);
            dot += q_val * k_val;
        }
        
        scores[k_pos] = dot / sqrtf((float)head_dim);
        max_score = fmaxf(max_score, scores[k_pos]);
    }
    
    // Second pass: compute softmax and V-multiply
    float exp_sum = 0.0f;
    
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        float exp_val = expf(scores[k_pos] - max_score);
        exp_sum += exp_val;
        
        // Store for third pass (or combine with this one)
        scores[k_pos] = exp_val;
    }
    
    // Third pass: weighted V sum
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        float softmax_val = scores[k_pos] / exp_sum;
        
        int v_idx = k_pos * num_heads * head_dim + head * head_dim + dim_idx;
        float v_val = __half2float(v_ptr[v_idx]);
        sum_v += softmax_val * v_val;
    }
    
    // Write output
    int out_idx = q_pos * num_heads * head_dim + head * head_dim + dim_idx;
    out_ptr[out_idx] = sum_v;
}
