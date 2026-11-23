// Versão de CPU: mini-batch com OpenMP. A parte comum está em barenn.h.
#include "barenn.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }

    try {
        const auto [treino, teste] = Barenn::carregar_dados(argc, argv);
        constexpr int epocas = Barenn::epocas;
        const int lote = Barenn::lote_configurado();
        const float taxa = Barenn::taxa_configurada();
        int threads = 1;
#ifdef _OPENMP
        threads = omp_get_max_threads();
#endif
        // Vai para stderr para não alterar a saída comparada na validação.
        cerr << "configuracao: lote " << lote << " | taxa " << taxa
             << " | threads " << threads << "\n";

        Barenn::Rede rede;
        const size_t P = rede.parametros.size();
        const int n = static_cast<int>(treino.size());
        // Gradiente de cada amostra do lote, um bloco de P floats por amostra.
        vector<float> grad_lote(static_cast<size_t>(lote) * P);
        vector<float> perda_amostra(n);
        array<double, epocas> perdas{};
        // A atualização é dividida entre as threads em faixas de parâmetros.
        constexpr size_t faixa = 256;
        const size_t faixas = (P + faixa - 1) / faixa;

        // A medição inclui forward, perda, backward e atualização de todas as
        // épocas.
        const auto inicio = chrono::steady_clock::now();
#pragma omp parallel
        {
            Barenn::Trabalho t;
            for (int epoca = 0; epoca < epocas; ++epoca) {
                // Uma época percorre todo o conjunto na ordem do arquivo.
                for (int ini = 0; ini < n; ini += lote) {
                    const int b = min(lote, n - ini);
                    // Gradientes das amostras do lote, em paralelo. Os pesos
                    // não mudam nesta fase.
#pragma omp for schedule(static)
                    for (int k = 0; k < b; ++k) {
                        const auto& amostra = treino[ini + k];
                        perda_amostra[ini + k] = Barenn::gradiente_amostra(
                            rede, amostra.imagem.data(), amostra.rotulo, t,
                            grad_lote.data() + static_cast<size_t>(k) * P);
                    }
                    // Atualização: cada thread cuida de faixas de parâmetros e
                    // soma os gradientes do lote sempre na ordem das amostras,
                    // então o resultado não depende do número de threads.
                    const float passo = taxa / b;
#pragma omp for schedule(static)
                    for (size_t f = 0; f < faixas; ++f) {
                        const size_t q0 = f * faixa;
                        const size_t q1 = min(P, q0 + faixa);
                        float soma[faixa];
                        const float* g0 = grad_lote.data();
                        for (size_t q = q0; q < q1; ++q) {
                            soma[q - q0] = g0[q];
                        }
                        for (int k = 1; k < b; ++k) {
                            const float* gk =
                                grad_lote.data() + static_cast<size_t>(k) * P;
                            for (size_t q = q0; q < q1; ++q) {
                                soma[q - q0] += gk[q];
                            }
                        }
                        float* p = rede.parametros.data();
                        for (size_t q = q0; q < q1; ++q) {
                            p[q] -= passo * soma[q - q0];
                        }
                    }
                }
#pragma omp single
                {
                    double perda_total = 0;
                    for (int s = 0; s < n; ++s) {
                        perda_total += perda_amostra[s];
                    }
                    perdas[epoca] = perda_total / treino.size();
                }
            }
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
        Barenn::imprimir_treino(treino.size(), perdas.data(), tempo.count());
        Barenn::avaliar(rede, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
