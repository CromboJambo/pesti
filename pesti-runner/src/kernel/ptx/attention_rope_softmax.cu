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

// Outputs per-head RAW scores (no causal mask, no scaling)
// Grid: (ceil(seq_q/128), seq_k, num_heads)
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

    // Output: per-head raw dot product (no mask, no scale)
    int out_idx = q_pos * num_heads * seq_k + head * seq_k + k_pos;
    s_ptr[out_idx] = dot_product;
}
