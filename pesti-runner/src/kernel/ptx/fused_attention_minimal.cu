//! Ultra-minimal kernel with NO local arrays - everything in registers

#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

__global__ void fused_attention_minimal_kernel(
    const half* __restrict__ q_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    half* __restrict__ out_ptr,
    float scale,
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim
) {
    int q_pos = blockIdx.x;
    int head = blockIdx.y;
    int d = threadIdx.x;
    
    if (q_pos >= seq_q || head >= num_heads || d >= head_dim) return;
    
    // ========================================================================
    // Compute attention scores and softmax weights in ONE pass (no arrays!)
    // ========================================================================
    
    float max_score = -FLT_MAX;
    
    // First pass: find max score
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        if (k_pos > q_pos) continue;  // Causal mask
        
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; dim++) {
            int idx_q = q_pos * num_heads * head_dim + head * head_dim + dim;
            int idx_k = k_pos * num_heads * head_dim + head * head_dim + dim;
            
            float q_val = __half2float(q_ptr[idx_q]);
            float k_val = __half2float(k_ptr[idx_k]);
            score += q_val * k_val;
        }
        
        score *= scale;
        if (score > max_score) {
            max_score = score;
        }
    }
    
    // Second pass: compute softmax weights and weighted V sum
    float exp_sum = 0.0f;
    float out_val = 0.0f;
    
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        if (k_pos > q_pos) continue;  // Causal mask
        
        // Recompute score (no storage, just recompute)
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; dim++) {
            int idx_q = q_pos * num_heads * head_dim + head * head_dim + dim;
            int idx_k = k_pos * num_heads * head_dim + head * head_dim + dim;
            
            float q_val = __half2float(q_ptr[idx_q]);
            float k_val = __half2float(k_ptr[idx_k]);
            score += q_val * k_val;
        }
        
        score *= scale;
        
        // Compute softmax weight
        float exp_val = expf(score - max_score);
        exp_sum += exp_val;
    }
    
    // Third pass: compute weighted V sum using softmax weights
    for (int k_pos = 0; k_pos < seq_k; k_pos++) {
        if (k_pos > q_pos) continue;  // Causal mask
        
        // Recompute score again
        float score = 0.0f;
        for (int dim = 0; dim < head_dim; dim++) {
            int idx_q = q_pos * num_heads * head_dim + head * head_dim + dim;
            int idx_k = k_pos * num_heads * head_dim + head * head_dim + dim;
            
            float q_val = __half2float(q_ptr[idx_q]);
            float k_val = __half2float(k_ptr[idx_k]);
            score += q_val * k_val;
        }
        
        score *= scale;
        float exp_val = expf(score - max_score);
        float softmax_weight = exp_val / exp_sum;
        
        // Weighted V sum for THIS dimension
        int idx_v = k_pos * num_heads * head_dim + head * head_dim + d;
        float v_val = __half2float(v_ptr[idx_v]);
        out_val += softmax_weight * v_val;
    }
    
    // Store output
    int out_idx = q_pos * num_heads * head_dim + head * head_dim + d;
    out_ptr[out_idx] = __float2half(out_val);
}
