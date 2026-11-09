//! Ultra minimal f16 copy kernel: out[i] = q[i] for i < size.
//!
//! Compiled to PTX with: nvcc -arch=sm_89 -ptx copy_kernel.cu -o copy_kernel.ptx
//! Used by tests/ultra_minimal_test.rs to exercise the PTX -> module -> function
//! -> launch -> readback path with a trivially-verifiable kernel.

#include <cuda_fp16.h>

__global__ void copy_kernel(const unsigned short* __restrict__ q,
                            unsigned short* __restrict__ out,
                            int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        out[i] = q[i];
    }
}
