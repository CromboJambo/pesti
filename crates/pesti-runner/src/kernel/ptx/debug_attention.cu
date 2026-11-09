//! Debug kernel with print statements

#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

// Kernel 1: Compute raw attention scores
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
    
    for (int chunk = threadIdx.x; chunk < head_dim / 2; chunk += blockDim.x) {
        int d = chunk * 2;
        
        int q_idx = q_pos * num_heads * head_dim + head * head_dim + d;
        float q0 = __half2float(q_ptr[q_idx]);
        float q1 = __half2float(q_ptr[q_idx + 1]);
        
        int k_idx = k_pos * num_heads * head_dim + head * head_dim + d;
        float k0 = __half2float(k_ptr[k_idx]);
        float k1 = __half2float(k_ptr[k_idx + 1]);
        
        dot_product += q0 * k0 + q1 * k1;
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

// Kernel 2: Apply softmax AND multiply by V
__global__ void apply_softmax_and_output_kernel(
    float* __restrict__ scores,
    const half* __restrict__ v_ptr,
    int seq_q,
    int seq_k,
    int num_heads,
    int head_dim
) {
    int q_pos = blockIdx.x;
    int head = blockIdx.y;
    
    if (q_pos >= seq_q || head >= num_heads) return;
    
    int tid = threadIdx.x;
    
    // Thread 0: find max and compute softmax sum
    if (tid == 0) {
        float max_val = -INFINITY;
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            if (scores[idx] > max_val) {
                max_val = scores[idx];
            }
        }
        
        float exp_sum = 0.0f;
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            float val = scores[idx];
            float exp_val = (val == -INFINITY) ? 0.0f : expf(val - max_val);
            scores[idx] = exp_val;
            exp_sum += exp_val;
        }
        
        int max_idx = q_pos * num_heads * seq_k + head * seq_k;
        scores[max_idx] = exp_sum;
    }
    
    __syncthreads();
    
    float exp_sum = scores[q_pos * num_heads * seq_k + head * seq_k];
    
    if (tid == 0 && exp_sum > 0) {
        for (int k = 0; k < seq_k; k++) {
            int idx = q_pos * num_heads * seq_k + head * seq_k + k;
            scores[idx] /= exp_sum;
        }
    }
    
    __syncthreads();
    
    // Each thread computes one dimension of the output for this (q_pos, head) pair
    int dim_idx = tid * 2;
    if (dim_idx < head_dim) {
        float output_val = 0.0f;
        
        for (int k = 0; k < seq_k; k++) {
            int score_idx = q_pos * num_heads * seq_k + head * seq_k + k;
            float softmax_val = scores[score_idx];
            
            int v_idx = k * num_heads * head_dim + head * head_dim + dim_idx;
            float v0 = __half2float(v_ptr[v_idx]);
            if (dim_idx + 1 < head_dim) {
                float v1 = __half2float(v_ptr[v_idx + 1]);
                output_val += softmax_val * (v0 + v1);
            } else {
                output_val += softmax_val * v0;
            }
        }
        
        int out_idx = q_pos * num_heads * head_dim + head * head_dim + dim_idx;
        scores[out_idx] = output_val;
    }
}

__global__ void debug_kernel(
    const half* __restrict__ v_ptr,
    float* __restrict__ debug_out,
    int seq_k,
    int num_heads,
    int head_dim
) {
    int tid = threadIdx.x;
    
    if (tid < 4) {
        // Read V for k=0, head=0, dim=tid
        int v_idx = 0 * num_heads * head_dim + 0 * head_dim + tid;
        float v_val = __half2float(v_ptr[v_idx]);
        
        debug_out[tid] = v_val;
    }
}
