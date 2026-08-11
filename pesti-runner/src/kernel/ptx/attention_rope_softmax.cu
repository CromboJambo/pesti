#include <cuda_fp16.h>
#include <math.h>

__device__ __forceinline__ void apply_rope_pair(
    float& q0, float& q1,
    float cos_val, float sin_val
) {
    float new_q0 = q0 * cos_val - q1 * sin_val;
    float new_q1 = q0 * sin_val + q1 * cos_val;
    q0 = new_q0;
    q1 = new_q1;
}

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
    int q_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int k_pos = blockIdx.y;
    int head = blockIdx.z;

    if (q_pos >= seq_q || k_pos >= seq_k) return;

    float half_dim = head_dim / 2.0f;
    float dot_product = 0.0f;

    for (int chunk = 0; chunk < head_dim / 2; chunk++) {
        int d = chunk * 2;

        int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
        float q0 = __half2float(q_ptr[q_idx]);
        float q1 = __half2float(q_ptr[q_idx + 1]);

        float inv_freq = 1.0f / powf(rope_base, (float)chunk / half_dim);
        float freq = q_pos * inv_freq;
        float c = cosf(freq);
        float s = sinf(freq);
        apply_rope_pair(q0, q1, c, s);

        int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
        float k0 = __half2float(k_ptr[k_idx]);
        float k1 = __half2float(k_ptr[k_idx + 1]);

        float freq_k = k_pos * inv_freq;
        float c_k = cosf(freq_k);
        float s_k = sinf(freq_k);
        apply_rope_pair(k0, k1, c_k, s_k);

        dot_product += q0 * k0 + q1 * k1;
    }

    // Apply scale (1/sqrt(head_dim))
    dot_product *= scale;

    // Causal mask: set to -inf if q_pos >= k_pos
    if (q_pos >= k_pos) {
        dot_product = -INFINITY;
    }

    // Store raw score
    int out_idx = q_pos * num_heads * seq_k + head * seq_k + k_pos;
    s_ptr[out_idx] = dot_product;
}

// Kernel 2: Apply softmax per (q_pos, head) pair - thread 0 does all work
__global__ void apply_softmax_kernel(
    float* s_ptr,
    int seq_q,
    int seq_k,
    int num_heads
) {
    int q_pos = blockIdx.x;
    int head = blockIdx.y;

    if (q_pos >= seq_q || head >= num_heads) return;

    // Thread 0 does all softmax computation for this (q_pos, head) pair
    if (threadIdx.x == 0) {
        // Find max
        float max_val = -INFINITY;
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            if (s_ptr[idx] > max_val) {
                max_val = s_ptr[idx];
            }
        }

        // Compute exp and sum
        float exp_sum = 0.0f;
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            float val = s_ptr[idx];
            float exp_val = (val == -INFINITY) ? 0.0f : expf(val - max_val);
            s_ptr[idx] = exp_val;  // Store exp value
            exp_sum += exp_val;
        }

        // Normalize
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            s_ptr[idx] /= exp_sum;
        }
    }
}
