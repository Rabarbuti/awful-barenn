// Versão de GPU (CUDA): o mesmo mini-batch da versão de CPU. Por lote, um
// kernel faz forward, perda e backward de todas as amostras (um bloco de
// threads por amostra, dados intermediários em memória compartilhada) e outro
// atualiza os pesos. A parte comum (rede, dados, avaliação) está em barenn.h.
#include "barenn.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define INDICE() static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)
// Lança um kernel com uma thread por item; o kernel recebe o total de itens.
#define LANCAR(kernel, total, ...) \
    kernel<<<((total) + kThreads - 1) / kThreads, kThreads>>>((total), __VA_ARGS__)
// Kernel de amostras: um bloco de kThreadsAmostra threads por amostra.
#define THREAD() static_cast<int>(threadIdx.x)
#define AMOSTRA() static_cast<int>(blockIdx.x)
#define LANCAR_AMOSTRAS(kernel, blocos, ...) \
    kernel<<<(blocos), kThreadsAmostra>>>(__VA_ARGS__)
// Uma fase do kernel de amostras, seguida da sincronização do bloco.
#define FASE(chamada)    \
    do {                 \
        chamada;         \
        __syncthreads(); \
    } while (0)
#else
// Sem nvcc, os kernels rodam em laço na CPU: serve só para testar a lógica.
#include "tests/cuda_emulado.h"
#endif

namespace {

constexpr int kThreads = 128;
// Threads por amostra: uma por entrada, o que cobre a maior camada.
constexpr int kThreadsAmostra = Barenn::entradas;

void checar(cudaError_t erro, const char* chamada) {
    if (erro != cudaSuccess) {
        throw runtime_error(string("CUDA: ") + cudaGetErrorString(erro) +
                            " em " + chamada);
    }
}
#define CUDA_OK(chamada) checar((chamada), #chamada)

// Memória na GPU, liberada automaticamente no fim.
template <typename T>
struct Buffer {
    T* ptr = nullptr;
    explicit Buffer(size_t n) { CUDA_OK(cudaMalloc(&ptr, n * sizeof(T))); }
    ~Buffer() { cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

// Camada como os kernels a enxergam: dimensões e posição dos pesos e do
// viés no vetor de parâmetros (mesmo layout transposto da CPU).
struct CamadaGPU {
    int entradas;
    int saidas;
    int final;
    int pesos;
    int bias;
};

CamadaGPU para_gpu(const Barenn::Camada& c) {
    return {c.in_features, c.out_features, c.final,
            static_cast<int>(c.inicio),
            static_cast<int>(c.inicio + c.num_pesos())};
}

// Dados de uma amostra na memória compartilhada do bloco: a entrada e, para
// cada camada, z, ativação, delta e gradiente da entrada.
struct Compartilhado {
    float x[Barenn::entradas];
    float z1[Barenn::ocultos1], a1[Barenn::ocultos1];
    float d1[Barenn::ocultos1], ga1[Barenn::ocultos1];
    float z2[Barenn::ocultos2], a2[Barenn::ocultos2];
    float d2[Barenn::ocultos2], ga2[Barenn::ocultos2];
    float z3[Barenn::classes], escores[Barenn::classes];
    float d3[Barenn::classes], ge[Barenn::classes];
};

// O que as fases precisam: camadas, pesos, dados da amostra e onde gravar
// a perda e o gradiente dela.
struct Contexto {
    CamadaGPU c1, c2, c3;
    const float* p;
    const float* imagem;
    int rotulo;
    float* perda;
    float* g;
    Compartilhado* s;
};

// z = viés + soma das entradas vezes os pesos, na ordem das entradas, como
// na CPU; ReLU nas camadas ocultas, igual a max(0.0f, z).
__device__ inline void forward_neuronio(const CamadaGPU& c, const float* p,
                                        const float* x, int i, float* z,
                                        float* a) {
    float s = p[c.bias + i];
    for (int j = 0; j < c.entradas; ++j) {
        s += x[j] * p[c.pesos + j * c.saidas + i];
    }
    z[i] = s;
    a[i] = c.final ? s : (0.0f < s ? s : 0.0f);
}

// Delta: gradiente em relação a z (na ReLU, só passa se z > 0).
__device__ inline void delta_neuronio(const CamadaGPU& c, const float* grad,
                                      const float* z, int i, float* delta) {
    delta[i] = c.final ? grad[i] : grad[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
}

// Gradiente da entrada j, para a camada anterior.
__device__ inline float grad_entrada(const CamadaGPU& c, const float* p,
                                     const float* delta, int j) {
    float s = 0;
    for (int i = 0; i < c.saidas; ++i) {
        s += delta[i] * p[c.pesos + j * c.saidas + i];
    }
    return s;
}

// Gradiente dos parâmetros da camada para a amostra: as threads do bloco
// dividem os parâmetros entre si.
__device__ inline void grad_parametros(int t, const CamadaGPU& c,
                                       const float* x, const float* delta,
                                       float* g) {
    const int num_pesos = c.entradas * c.saidas;
    for (int q = t; q < num_pesos + c.saidas; q += kThreadsAmostra) {
        if (q < num_pesos) {
            const int j = q / c.saidas;
            const int i = q % c.saidas;
            g[c.pesos + q] = delta[i] * x[j];
        } else {
            const int i = q - num_pesos;
            g[c.bias + i] = delta[i];
        }
    }
}

// Fases do kernel de amostras. Entre uma fase e a seguinte, o bloco se
// sincroniza, então cada fase só lê o que as anteriores já gravaram na
// memória compartilhada. t é o índice da thread no bloco.
__device__ inline void fase_carregar(int t, const Contexto& ctx) {
    if (t < Barenn::entradas) ctx.s->x[t] = ctx.imagem[t];
}
__device__ inline void fase_forward1(int t, const Contexto& ctx) {
    Compartilhado& s = *ctx.s;
    if (t < ctx.c1.saidas) forward_neuronio(ctx.c1, ctx.p, s.x, t, s.z1, s.a1);
}
__device__ inline void fase_forward2(int t, const Contexto& ctx) {
    Compartilhado& s = *ctx.s;
    if (t < ctx.c2.saidas) forward_neuronio(ctx.c2, ctx.p, s.a1, t, s.z2, s.a2);
}
__device__ inline void fase_forward3(int t, const Contexto& ctx) {
    Compartilhado& s = *ctx.s;
    if (t < ctx.c3.saidas) {
        forward_neuronio(ctx.c3, ctx.p, s.a2, t, s.z3, s.escores);
    }
}
// Perda e gradiente dos escores, somados na ordem das classes, como na CPU.
__device__ inline void fase_perda(int t, const Contexto& ctx) {
    if (t != 0) return;
    Compartilhado& s = *ctx.s;
    float soma = 0;
    for (int i = 0; i < Barenn::classes; ++i) {
        const float erro = s.escores[i] - (i == ctx.rotulo ? 1.0f : 0.0f);
        soma += 0.5f * erro * erro;
        s.ge[i] = erro;
    }
    *ctx.perda = soma;
}
__device__ inline void fase_delta3(int t, const Contexto& ctx) {
    Compartilhado& s = *ctx.s;
    if (t < ctx.c3.saidas) delta_neuronio(ctx.c3, s.ge, s.z3, t, s.d3);
}
__device__ inline void fase_gradientes3(int t, const Contexto& ctx) {
    Compartilhado& s = *ctx.s;
    if (t < ctx.c3.entradas) s.ga2[t] = grad_entrada(ctx.c3, ctx.p, s.d3, t);
    grad_parametros(t, ctx.c3, s.a2, s.d3, ctx.g);
}
__device__ inline void fase_delta2(int t, const Contexto& ctx) {
    Compartilhado& s = *ctx.s;
    if (t < ctx.c2.saidas) delta_neuronio(ctx.c2, s.ga2, s.z2, t, s.d2);
}
__device__ inline void fase_gradientes2(int t, const Contexto& ctx) {
    Compartilhado& s = *ctx.s;
    if (t < ctx.c2.entradas) s.ga1[t] = grad_entrada(ctx.c2, ctx.p, s.d2, t);
    grad_parametros(t, ctx.c2, s.a1, s.d2, ctx.g);
}
__device__ inline void fase_delta1(int t, const Contexto& ctx) {
    Compartilhado& s = *ctx.s;
    if (t < ctx.c1.saidas) delta_neuronio(ctx.c1, s.ga1, s.z1, t, s.d1);
}
// A primeira camada não calcula o gradiente da entrada.
__device__ inline void fase_gradientes1(int t, const Contexto& ctx) {
    Compartilhado& s = *ctx.s;
    grad_parametros(t, ctx.c1, s.x, s.d1, ctx.g);
}

// Forward, perda e backward de todas as amostras do lote num só kernel: um
// bloco por amostra, uma thread por neurônio. Grava a perda da amostra k em
// perdas[k] e o gradiente dela em g + k * P.
__global__ void k_amostras(CamadaGPU c1, CamadaGPU c2, CamadaGPU c3, int P,
                           const float* p, const float* imagens,
                           const int* rotulos, float* perdas, float* g) {
    __shared__ Compartilhado s;
    const int k = AMOSTRA();
    const Contexto ctx{c1, c2, c3, p, imagens + k * Barenn::entradas,
                       rotulos[k], perdas + k, g + k * P, &s};
    FASE(fase_carregar(THREAD(), ctx));
    FASE(fase_forward1(THREAD(), ctx));
    FASE(fase_forward2(THREAD(), ctx));
    FASE(fase_forward3(THREAD(), ctx));
    FASE(fase_perda(THREAD(), ctx));
    FASE(fase_delta3(THREAD(), ctx));
    FASE(fase_gradientes3(THREAD(), ctx));
    FASE(fase_delta2(THREAD(), ctx));
    FASE(fase_gradientes2(THREAD(), ctx));
    FASE(fase_delta1(THREAD(), ctx));
    FASE(fase_gradientes1(THREAD(), ctx));
}

// Atualização: uma thread por parâmetro soma os gradientes do lote na ordem
// das amostras, como na CPU.
__global__ void k_atualizar(int total, int b, float passo, const float* g,
                            float* p) {
    const int q = INDICE();
    if (q >= total) return;
    float soma = g[q];
    for (int k = 1; k < b; ++k) {
        soma += g[k * total + q];
    }
    p[q] -= passo * soma;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }

    try {
        const auto [treino, teste] = Barenn::carregar_dados(argc, argv);
        constexpr int epocas = Barenn::epocas;
        constexpr int entradas = Barenn::entradas;
        const int lote = Barenn::lote_configurado();
        const float taxa = Barenn::taxa_configurada();

        cudaDeviceProp gpu{};
        CUDA_OK(cudaGetDeviceProperties(&gpu, 0));
        // Vai para stderr para não alterar a saída comparada na validação.
        cerr << "configuracao: lote " << lote << " | taxa " << taxa
             << " | gpu " << gpu.name << "\n";

        Barenn::Rede rede;
        const int P = static_cast<int>(rede.parametros.size());
        const int n = static_cast<int>(treino.size());
        const CamadaGPU c1 = para_gpu(rede.l1);
        const CamadaGPU c2 = para_gpu(rede.l2);
        const CamadaGPU c3 = para_gpu(rede.l3);

        // Imagens e rótulos contíguos, copiados para a GPU de uma vez.
        vector<float> imagens(static_cast<size_t>(n) * entradas);
        vector<int> rotulos(n);
        for (int s = 0; s < n; ++s) {
            copy(treino[s].imagem.begin(), treino[s].imagem.end(),
                 imagens.begin() + static_cast<size_t>(s) * entradas);
            rotulos[s] = treino[s].rotulo;
        }

        // Memória na GPU. A primeira alocação cria o contexto CUDA, que fica
        // fora do cronômetro.
        const size_t L = lote;
        // Conjunto de treino inteiro e a perda de cada amostra em cada época
        // ficam na GPU: nenhuma cópia dentro do laço de treino.
        const size_t perdas_total = static_cast<size_t>(epocas) * n;
        Buffer<float> d_p(P), d_g(L * P);
        Buffer<float> d_imagens(static_cast<size_t>(n) * entradas);
        Buffer<int> d_rotulos(n);
        Buffer<float> d_perdas(perdas_total);
        vector<float> perda_amostra(perdas_total);
        array<double, epocas> perdas{};

        // A medição inclui as cópias entre CPU e GPU, os kernels e a cópia
        // final dos pesos e das perdas de volta para a CPU.
        const auto inicio = chrono::steady_clock::now();
        CUDA_OK(cudaMemcpy(d_p.ptr, rede.parametros.data(), P * sizeof(float),
                           cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_imagens.ptr, imagens.data(),
                           imagens.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_OK(cudaMemcpy(d_rotulos.ptr, rotulos.data(), n * sizeof(int),
                           cudaMemcpyHostToDevice));
        // Sem cópias no laço, a CPU só enfileira os kernels, sem esperar a GPU.
        for (int epoca = 0; epoca < epocas; ++epoca) {
            for (int ini = 0; ini < n; ini += lote) {
                const int b = min(lote, n - ini);
                // Lote atual, direto do conjunto que já está na GPU.
                const float* d_x = d_imagens.ptr + static_cast<size_t>(ini) * entradas;
                const int* d_rot = d_rotulos.ptr + ini;
                float* d_perda_lote =
                    d_perdas.ptr + static_cast<size_t>(epoca) * n + ini;

                // Forward, perda e backward de todas as amostras do lote.
                LANCAR_AMOSTRAS(k_amostras, b, c1, c2, c3, P, d_p.ptr, d_x, d_rot,
                                d_perda_lote, d_g.ptr);

                // Atualização dos pesos com a soma dos gradientes do lote.
                const float passo = taxa / b;
                LANCAR(k_atualizar, P, b, passo, d_g.ptr, d_p.ptr);
                // Só consulta erros de lançamento; não espera a GPU.
                CUDA_OK(cudaGetLastError());
            }
        }
        // Única espera pela GPU: pesos e perdas de todas as épocas voltam no fim.
        CUDA_OK(cudaMemcpy(rede.parametros.data(), d_p.ptr, P * sizeof(float),
                           cudaMemcpyDeviceToHost));
        CUDA_OK(cudaMemcpy(perda_amostra.data(), d_perdas.ptr,
                           perdas_total * sizeof(float), cudaMemcpyDeviceToHost));
        for (int epoca = 0; epoca < epocas; ++epoca) {
            const float* pe = perda_amostra.data() + static_cast<size_t>(epoca) * n;
            double perda_total = 0;
            for (int s = 0; s < n; ++s) {
                perda_total += pe[s];
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // A avaliação roda na CPU, com os pesos treinados na GPU.
        Barenn::imprimir_treino(treino.size(), perdas.data(), tempo.count());
        Barenn::avaliar(rede, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
