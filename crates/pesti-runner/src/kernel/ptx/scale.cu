// Simple scale kernel: y[i] = x[i] * 2.0
extern "C" {
    __global__ void scale_kernel(float* x, float* y, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            y[i] = x[i] * 2.0f;
        }
    }
}