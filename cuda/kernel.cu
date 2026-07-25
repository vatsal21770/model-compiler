#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Fused bias + GELU (tanh approximation).
// y[i] = GELU(x[i] + bias[i % C])

__device__ __forceinline__ float gelu(float v) {
    const float k = 0.7978845608f;              // sqrt(2/pi)
    float inner = k * (v + 0.044715f * v * v * v);
    return 0.5f * v * (1.0f + tanhf(inner));
}

__global__ void fused_bias_gelu(const float* x, const float* bias,
                                 float* y, int n, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] + bias[i % C];
        y[i] = gelu(v);
    }
}

static void write_bin(const char* path, const float* data, int n) {
    FILE* f = fopen(path, "wb");
    fwrite(data, sizeof(float), n, f);
    fclose(f);
}

int main() {
    int N = 32;
    int C = 512;
    int n = N * C;
    size_t bytes_x = n * sizeof(float);
    size_t bytes_b = C * sizeof(float);

    float* h_x = (float*)malloc(bytes_x);
    float* h_b = (float*)malloc(bytes_b);
    float* h_y = (float*)malloc(bytes_x);

    // Deterministic pseudo-random inputs spanning negatives and positives.
    srand(0);
    for (int i = 0; i < n; i++) h_x[i] = -4.0f + 8.0f * ((float)rand() / (float)RAND_MAX);
    for (int c = 0; c < C; c++) h_b[c] = -1.0f + 2.0f * ((float)rand() / (float)RAND_MAX);

    float *d_x, *d_b, *d_y;
    cudaMalloc(&d_x, bytes_x);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_y, bytes_x);

    cudaMemcpy(d_x, h_x, bytes_x, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fused_bias_gelu<<<blocks, threads>>>(d_x, d_b, d_y, n, C);

    cudaError_t err = cudaGetLastError();           // catch launch failures
    if (err != cudaSuccess) {
        printf("kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_y, d_y, bytes_x, cudaMemcpyDeviceToHost);

    // Dump inputs and GPU output for the PyTorch cross-check.
    write_bin("x.bin", h_x, n);
    write_bin("bias.bin", h_b, C);
    write_bin("y_gpu.bin", h_y, n);

    printf("N=%d C=%d n=%d  |  %d blocks x %d threads\n", N, C, n, blocks, threads);
    printf("wrote x.bin, bias.bin, y_gpu.bin\n");

    cudaFree(d_x); cudaFree(d_b); cudaFree(d_y);
    free(h_x); free(h_b); free(h_y);
    return 0;
}
