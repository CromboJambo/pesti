// Attention concentration validation kernel for early-exit inference.
// Computes max attention weight across all positions for each head.
// High concentration = model confident, can potentially exit early.
//
// Input: attention weights (softmax output) [batch * heads * seq_len]
// Output: concentration scores [batch * heads] where score in [0, 1]

extern "C" __global__ void attention_concentration_kernel(
    const float* attention_weights,
    float* concentrations,
    int seq_len,
    int num_heads,
    int batch_size) {
    
    // Each thread block handles one (batch, head) pair
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    // Compute max attention weight for this (batch, head) pair
    const float* head_weights = attention_weights + 
        (batch_idx * num_heads + head_idx) * seq_len;
    
    float local_max = 0.0f;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        if (head_weights[i] > local_max) {
            local_max = head_weights[i];
        }
    }
    
    // Block-level max reduction via shared memory
    __shared__ float block_max[256];
    block_max[threadIdx.x] = local_max;
    __syncthreads();
    
    // Tree-based reduction across block
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float a = block_max[threadIdx.x];
            float b = block_max[threadIdx.x + stride];
            if (b > a) {
                block_max[threadIdx.x] = b;
            }
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (threadIdx.x == 0) {
        concentrations[batch_idx * num_heads + head_idx] = block_max[0];
    }
}

// Host API wrapper
extern "C" cudaError_t pesti_attention_concentration(
    const float* d_attention,
    float* d_concentrations,
    int batch_size,
    int num_heads,
    int seq_len,
    cudaStream_t stream) {
    
    dim3 grid(batch_size, num_heads);
    dim3 block(256);
    
    attention_concentration_kernel<<<grid, block, 0, stream>>>(
        d_attention, d_concentrations, seq_len, num_heads, batch_size);
    
    return cudaGetLastError();
}