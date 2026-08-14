//! Minimal fused attention kernel - scores only (matches exact_pattern approach)
//! Will be followed by softmax + V-multiply in a second kernel pass

#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

__global__ void fused_attention_simple_kernel(
    const half* __restrict__ q_ptr,      // [seq_q, num_heads, head_dim]
    const half* __restrict__ k_ptr,      // [seq_k, num_heads, head_dim]  
    const half* __restrict__ v_ptr,      // [seq_k, num_heads, head_dim] - not used in this kernel
    float* __restrict__ scores_ptr,       // [seq_q, num_heads, seq_k] - output scores
    int seq_q,                            // Query sequence length
    int seq_k,                            // Key/value sequence length
    int num_heads,                        // Number of attention heads
    int head_dim,                         // Dimension per head
    float scale                           // 1/sqrt(head_dim)
) {
    int q_pos = blockIdx.x;               // Query position
    int k_pos = blockIdx.y;               // Key position
    int head = blockIdx.z;                // Attention head
    int tid = threadIdx.x;                // Thread dimension
    
    if (q_pos >= seq_q || k_pos >= seq_k || head >= num_heads) return;
    
    // Compute dot product Q @ K for this (q_pos, k_pos, head)
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
    
    // Apply causal mask: mask out future tokens (k_pos > q_pos)
    if (k_pos > q_pos) {
        score = -FLT_MAX;
    }
    
    // Write score to output buffer
    int score_idx = q_pos * num_heads * seq_k + head * seq_k + k_pos;
    scores_ptr[score_idx] = score;
}
