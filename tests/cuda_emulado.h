// Emulação mínima da API CUDA na CPU, só para testar a lógica de
// barenn_cuda.cu sem GPU:
//
//   g++ -std=c++17 -x c++ barenn_cuda.cu -o build/barenn_cuda_emulado
//
// Cada kernel roda num laço comum, uma "thread" por vez. Não substitui o
// teste na GPU: serve para conferir a ordem das contas e os índices.
// O binário de GPU (nvcc) não usa este arquivo.
#pragma once

#include <cstdlib>
#include <cstring>

#define __global__

inline thread_local int indice_emulado = 0;
#define INDICE() (indice_emulado)
#define LANCAR(kernel, total, ...)                                      \
    for (indice_emulado = 0; indice_emulado < (total); ++indice_emulado) \
    kernel((total), __VA_ARGS__)

enum cudaError_t { cudaSuccess = 0 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
struct cudaDeviceProp {
    char name[256];
};

inline const char* cudaGetErrorString(cudaError_t) { return "erro emulado"; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) {
    std::strcpy(prop->name, "emulacao na CPU");
    return cudaSuccess;
}
template <typename T>
cudaError_t cudaMalloc(T** ptr, std::size_t bytes) {
    *ptr = static_cast<T*>(std::malloc(bytes));
    return cudaSuccess;
}
inline cudaError_t cudaFree(void* ptr) {
    std::free(ptr);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void* destino, const void* origem,
                              std::size_t bytes, cudaMemcpyKind) {
    std::memcpy(destino, origem, bytes);
    return cudaSuccess;
}
