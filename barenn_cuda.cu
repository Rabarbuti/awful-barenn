// Versão de GPU (CUDA): o mesmo mini-batch da versão de CPU, com um kernel
// por etapa de cada camada e uma thread por neurônio de cada amostra.
// A parte comum (rede, dados, avaliação) está em barenn.h.
#include "barenn.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define INDICE() static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)
// Lança um kernel com uma thread por item; o kernel recebe o total de itens.
#define LANCAR(kernel, total, ...) \
    kernel<<<((total) + kThreads - 1) / kThreads, kThreads>>>((total), __VA_ARGS__)
#else
// Sem nvcc, os kernels rodam em laço na CPU: serve só para testar a lógica.
#include "tests/cuda_emulado.h"
#endif

namespace {

constexpr int kThreads = 128;

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

// z = viés + soma das entradas vezes os pesos; uma thread por neurônio de
// cada amostra. A soma segue a ordem das entradas, como na CPU.
__global__ void k_forward(int total, CamadaGPU c, const float* p,
                          const float* x, float* z, float* a) {
    const int t = INDICE();
    if (t >= total) return;
    const int k = t / c.saidas;
    const int i = t % c.saidas;
    const float* xk = x + k * c.entradas;
    float s = p[c.bias + i];
    for (int j = 0; j < c.entradas; ++j) {
        s += xk[j] * p[c.pesos + j * c.saidas + i];
    }
    z[t] = s;
    // ReLU nas camadas ocultas, igual a max(0.0f, s) da CPU.
    a[t] = c.final ? s : (0.0f < s ? s : 0.0f);
}

// Perda da amostra e gradiente em relação aos escores; uma thread por amostra.
__global__ void k_perda(int total, int classes, const float* escores,
                        const int* rotulos, float* ge, float* perdas) {
    const int k = INDICE();
    if (k >= total) return;
    float soma = 0;
    for (int i = 0; i < classes; ++i) {
        const float erro =
            escores[k * classes + i] - (i == rotulos[k] ? 1.0f : 0.0f);
        soma += 0.5f * erro * erro;
        ge[k * classes + i] = erro;
    }
    perdas[k] = soma;
}

// Delta: gradiente em relação a z (na ReLU, só passa se z > 0).
__global__ void k_delta(int total, CamadaGPU c, const float* grad_saida,
                        const float* z, float* delta) {
    const int t = INDICE();
    if (t >= total) return;
    delta[t] = c.final ? grad_saida[t]
                       : grad_saida[t] * (z[t] > 0.0f ? 1.0f : 0.0f);
}

// Gradiente da entrada, para a camada anterior; uma thread por entrada de
// cada amostra.
__global__ void k_grad_entrada(int total, CamadaGPU c, const float* p,
                               const float* delta, float* grad_entrada) {
    const int t = INDICE();
    if (t >= total) return;
    const int k = t / c.entradas;
    const int j = t % c.entradas;
    const float* dk = delta + k * c.saidas;
    float s = 0;
    for (int i = 0; i < c.saidas; ++i) {
        s += dk[i] * p[c.pesos + j * c.saidas + i];
    }
    grad_entrada[t] = s;
}

// Gradiente dos parâmetros da camada para cada amostra; uma thread por
// parâmetro de cada amostra. O gradiente da amostra k fica em g + k * P.
__global__ void k_grad_parametros(int total, CamadaGPU c, int P,
                                  const float* x, const float* delta,
                                  float* g) {
    const int t = INDICE();
    if (t >= total) return;
    const int num_pesos = c.entradas * c.saidas;
    const int por_amostra = num_pesos + c.saidas;
    const int k = t / por_amostra;
    const int q = t % por_amostra;
    float* gk = g + k * P;
    if (q < num_pesos) {
        const int j = q / c.saidas;
        const int i = q % c.saidas;
        gk[c.pesos + q] = delta[k * c.saidas + i] * x[k * c.entradas + j];
    } else {
        const int i = q - num_pesos;
        gk[c.bias + i] = delta[k * c.saidas + i];
    }
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
        constexpr int classes = Barenn::classes;
        constexpr int o1 = Barenn::ocultos1;
        constexpr int o2 = Barenn::ocultos2;
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
        const int p1 = c1.entradas * c1.saidas + c1.saidas;
        const int p2 = c2.entradas * c2.saidas + c2.saidas;
        const int p3 = c3.entradas * c3.saidas + c3.saidas;

        // Imagens e rótulos contíguos, para copiar um lote de cada vez.
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
        Buffer<float> d_p(P), d_g(L * P), d_x(L * entradas);
        Buffer<int> d_rotulos(L);
        Buffer<float> d_z1(L * o1), d_a1(L * o1), d_d1(L * o1), d_ga1(L * o1);
        Buffer<float> d_z2(L * o2), d_a2(L * o2), d_d2(L * o2), d_ga2(L * o2);
        Buffer<float> d_z3(L * classes), d_escores(L * classes);
        Buffer<float> d_d3(L * classes), d_ge(L * classes), d_perdas(L);
        vector<float> perda_amostra(n);
        array<double, epocas> perdas{};

        // A medição inclui as cópias entre CPU e GPU, os kernels e a cópia
        // final dos pesos de volta para a CPU.
        const auto inicio = chrono::steady_clock::now();
        CUDA_OK(cudaMemcpy(d_p.ptr, rede.parametros.data(), P * sizeof(float),
                           cudaMemcpyHostToDevice));
        for (int epoca = 0; epoca < epocas; ++epoca) {
            for (int ini = 0; ini < n; ini += lote) {
                const int b = min(lote, n - ini);
                CUDA_OK(cudaMemcpy(d_x.ptr,
                                   imagens.data() + static_cast<size_t>(ini) * entradas,
                                   static_cast<size_t>(b) * entradas * sizeof(float),
                                   cudaMemcpyHostToDevice));
                CUDA_OK(cudaMemcpy(d_rotulos.ptr, rotulos.data() + ini,
                                   b * sizeof(int), cudaMemcpyHostToDevice));

                // Forward e perda.
                LANCAR(k_forward, b * o1, c1, d_p.ptr, d_x.ptr, d_z1.ptr, d_a1.ptr);
                LANCAR(k_forward, b * o2, c2, d_p.ptr, d_a1.ptr, d_z2.ptr, d_a2.ptr);
                LANCAR(k_forward, b * classes, c3, d_p.ptr, d_a2.ptr, d_z3.ptr,
                       d_escores.ptr);
                LANCAR(k_perda, b, classes, d_escores.ptr, d_rotulos.ptr, d_ge.ptr,
                       d_perdas.ptr);

                // Backward da última camada até a primeira.
                LANCAR(k_delta, b * classes, c3, d_ge.ptr, d_z3.ptr, d_d3.ptr);
                LANCAR(k_grad_entrada, b * o2, c3, d_p.ptr, d_d3.ptr, d_ga2.ptr);
                LANCAR(k_grad_parametros, b * p3, c3, P, d_a2.ptr, d_d3.ptr, d_g.ptr);
                LANCAR(k_delta, b * o2, c2, d_ga2.ptr, d_z2.ptr, d_d2.ptr);
                LANCAR(k_grad_entrada, b * o1, c2, d_p.ptr, d_d2.ptr, d_ga1.ptr);
                LANCAR(k_grad_parametros, b * p2, c2, P, d_a1.ptr, d_d2.ptr, d_g.ptr);
                // A primeira camada não calcula o gradiente da entrada.
                LANCAR(k_delta, b * o1, c1, d_ga1.ptr, d_z1.ptr, d_d1.ptr);
                LANCAR(k_grad_parametros, b * p1, c1, P, d_x.ptr, d_d1.ptr, d_g.ptr);

                // Atualização dos pesos com a soma dos gradientes do lote.
                const float passo = taxa / b;
                LANCAR(k_atualizar, P, b, passo, d_g.ptr, d_p.ptr);
                CUDA_OK(cudaGetLastError());

                CUDA_OK(cudaMemcpy(perda_amostra.data() + ini, d_perdas.ptr,
                                   b * sizeof(float), cudaMemcpyDeviceToHost));
            }
            double perda_total = 0;
            for (int s = 0; s < n; ++s) {
                perda_total += perda_amostra[s];
            }
            perdas[epoca] = perda_total / treino.size();
        }
        CUDA_OK(cudaMemcpy(rede.parametros.data(), d_p.ptr, P * sizeof(float),
                           cudaMemcpyDeviceToHost));
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // A avaliação roda na CPU, com os pesos treinados na GPU.
        Barenn::imprimir_treino(treino.size(), perdas.data(), tempo.count());
        Barenn::avaliar(rede, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
