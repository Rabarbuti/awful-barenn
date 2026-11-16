#include <cstdio>

__global__ void soma(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

int main() {
    const int n = 1 << 20;
    float* d;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));
    soma<<<(n + 255) / 256, 256>>>(d, n);
    cudaError_t e = cudaDeviceSynchronize();
    printf("kernel: %s\n", cudaGetErrorString(e));
    cudaFree(d);
    return e != cudaSuccess;
}
