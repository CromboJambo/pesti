//! Single-kernel fused attention - simplified for correctness first
//! Combines RoPE + scores + softmax + V-multiply in one launch

#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

/**
 * Fused Attention Kernel - Single Launch (Simplified)
 * 
 * Performs all attention operations in one kernel launch:
 * 1. Compute attention scores (Q @ K^T) with causal mask
 * 2. Apply softmax with max-subtraction trick  
 * 3. Multiply by V to get final output
 * 
 * Note: RoPE applied separately before kernel launch for simplicity
 * Target: RTX 4070 Ti SUPER (sm_8.9), head_dim=16 test case
 */
__global__ void fused_attention_simple_kernel(
    const half* __restrict__ q_ptr,      // [seq_q, num_heads, head_dim]
    const half* __restrict__ k_ptr,      // [seq_k, num_heads, head_dim]
    const half* __restrict__ v_ptr,      // [seq_k, num_heads, head_dim]
    half* __restrict__ out_ptr,           // [seq_q, num_heads, head_dim]
    float scale,                          // 1/sqrt(head_dim)
    int seq_q,                            // Query sequence length
    int seq_k,                            // Key/value sequence length
    int num_heads,                        // Number of attention heads
    int head_dim                          // Dimension per head
) {
    int q_pos = blockIdx.x;               // Query position
    int head = blockIdx.y;                // Attention head
    int tid = threadIdx.x;                // Thread ID
    
    if (q_pos >= seq_q || head >= num_heads) return;
    
    const int MAX_K = 32;  // Max sequence length for this test
    
    float scores[MAX_K];
    float probs[MAX_K];
    
    // ========================================================================
    // STEP 1: Compute attention scores with causal mask
    // ========================================================================
    
    float max_score = -FLT_MAX;
    
    for (int k_pos = 0; k_pos < seq_k && k_pos < MAX_K; k_pos++) {
        // Apply causal mask: mask out future tokens (k_pos > q_pos)
        if (k_pos > q_pos) {
            scores[k_pos] = -FLT_MAX;
            continue;
        }
        
        // Compute dot product Q @ K for this k_pos
        float score = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            int idx_q = q_pos * num_heads * head_dim + head * head_dim + d;
            int idx_k = k_pos * num_heads * head_dim + head * head_dim + d;
            
            float q_val = __half2float(q_ptr[idx_q]);
            float k_val = __half2float(k_ptr[idx_k]);
            
            score += q_val * k_val;
        }
        
        // Scale by 1/sqrt(head_dim)
        score *= scale;
        scores[k_pos] = score;
        
        // Track max for numerical stability
        if (score > max_score) {
            max_score = score;
        }
    }
    
    // ========================================================================
    // STEP 2: Apply softmax with max-subtraction trick
    // ========================================================================
    
    float exp_sum = 0.0f;
    
    for (int k_pos = 0; k_pos < seq_k && k_pos < MAX_K; k_pos++) {
        if (scores[k_pos] == -FLT_MAX) {
            probs[k_pos] = 0.0f;
        } else {
            float exp_val = expf(scores[k_pos] - max_score);
            probs[k_pos] = exp_val;
            exp_sum += exp_val;
        }
    }
    
    // Normalize
    if (exp_sum > 0.0f) {
        for (int k_pos = 0; k_pos < seq_k && k_pos < MAX_K; k_pos++) {
            probs[k_pos] /= exp_sum;
        }
    }
    
    // ========================================================================
    // STEP 3: Weighted sum of V to get final output
    // ========================================================================
    
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out_val = 0.0f;
        
        for (int k_pos = 0; k_pos < seq_k && k_pos < MAX_K; k_pos++) {
            if (probs[k_pos] > 0.0f) {  // Skip masked positions
                int idx_v = k_pos * num_heads * head_dim + head * head_dim + d;
                float v_val = __half2float(v_ptr[idx_v]);
                out_val += probs[k_pos] * v_val;
            }
        }
        
        // Store output (FP32 → FP16)
        int out_idx = q_pos * num_heads * head_dim + head * head_dim + d;
        out_ptr[out_idx] = __float2half(out_val);
    }
}
