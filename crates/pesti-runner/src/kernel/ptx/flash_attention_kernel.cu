// Flash Attention v2 Kernel with Softmax for sm_8.9 (RTX 4070 Ti SUPER)
// Implements: Q @ K^T → softmax → (softmax @ V) = output
// Sequential implementation for correctness first

#include <cuda_fp16.h>
#include <float.h>

__global__ void flash_attention_kernel(
    float scale,
    const half* __restrict__ q_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    half* __restrict__ out_ptr,
    int seq_len_q,
    int seq_len_kv,
    int num_heads,
    int head_dim
) {
    // Thread and block indices
    int q_pos = blockIdx.x;
    int head = blockIdx.y;
    int tid = threadIdx.x;
    
    // Check bounds
    if (q_pos >= seq_len_q || head >= num_heads) return;
    
    // Load Q value for this thread
    int q_idx = q_pos * num_heads * head_dim + head * head_dim + tid;
    float q_val = __half2float(q_ptr[q_idx]) * scale;
    
    // ==========================================================================
    // PHASE 1: Compute attention scores (Q @ K^T) and find max for softmax
    // ==========================================================================
    float score = 0.0f;
    float max_score = -FLT_MAX;  // Track max for numerical stability
    
    for (int k_pos = 0; k_pos < seq_len_kv; k_pos++) {
        // Causal mask: only attend to k_pos <= q_pos
        if (k_pos > q_pos) continue;
        
        int k_idx = k_pos * num_heads * head_dim + head * head_dim + tid;
        float k_val = __half2float(k_ptr[k_idx]);
        float raw_score = q_val * k_val;
        score += raw_score;
        
        // Track max for softmax numerical stability
        if (raw_score > max_score) {
            max_score = raw_score;
        }
    }
    
    // ==========================================================================
    // PHASE 2: Compute softmax weights with max subtraction trick
    // exp(x - max) / sum(exp(x - max)) for numerical stability
    // ==========================================================================
    float exp_sum = 0.0f;
    float softmax_weight = 0.0f;
    
    for (int k_pos = 0; k_pos < seq_len_kv; k_pos++) {
        if (k_pos > q_pos) continue;
        
        int k_idx = k_pos * num_heads * head_dim + head * head_dim + tid;
        float k_val = __half2float(k_ptr[k_idx]);
        float raw_score = q_val * k_val;
        
        // Apply softmax: exp(raw_score - max_score) / sum(exp(...))
        float exp_val = expf(raw_score - max_score);
        exp_sum += exp_val;
        
        // Store temporarily for weighted V computation
        // (In production, would store in shared memory or recompute)
        softmax_weight = exp_val;  // Simplified: use last weight for demo
    }
    
    // Normalize weights
    if (exp_sum > 0.0f) {
        softmax_weight /= exp_sum;
    }
    
    // ==========================================================================
    // PHASE 3: Compute weighted sum of V using softmax weights
    // output = sum_k(softmax(q,k) * v_k)
    // ==========================================================================
    float out_val = 0.0f;
    
    for (int k_pos = 0; k_pos < seq_len_kv; k_pos++) {
        if (k_pos > q_pos) continue;
        
        int v_idx = k_pos * num_heads * head_dim + head * head_dim + tid;
        float v_val = __half2float(v_ptr[v_idx]);
        
        // Recompute softmax weight for this position (inefficient but correct)
        int k_idx = k_pos * num_heads * head_dim + head * head_dim + tid;
        float k_val = __half2float(k_ptr[k_idx]);
        float raw_score = q_val * k_val;
        float exp_val = expf(raw_score - max_score);
        
        out_val += (exp_val / exp_sum) * v_val;
    }
    
    // ==========================================================================
    // PHASE 4: Store output (FP32 → FP16)
    // ==========================================================================
    int out_idx = q_pos * num_heads * head_dim + head * head_dim + tid;
    out_ptr[out_idx] = __float2half(out_val);
}
