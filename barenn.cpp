#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;
// Rede densa com dimensões 64 -> 32 -> 16 -> 10.
constexpr int ocultos1 = 32;
constexpr int ocultos2 = 16;

// Uma camada densa. Os parâmetros ficam no vetor único da rede
// (Rede::parametros), a partir de "inicio": primeiro os pesos, transpostos
// (peso[j * out_features + i] liga a entrada j ao neurônio i), depois o viés.
// Os métodos não guardam estado: os dados de cada amostra ficam em buffers
// de quem chama, então várias threads podem usar a mesma camada ao mesmo
// tempo.
struct Camada {
    int in_features;
    int out_features;
    bool final;
    // A primeira camada não tem camada anterior para receber o gradiente.
    bool primeira;
    size_t inicio;

    size_t num_pesos() const {
        return static_cast<size_t>(in_features) * out_features;
    }
    size_t num_parametros() const { return num_pesos() + out_features; }

    // z recebe a soma antes da ativação e a, a saída da camada.
    void Forward(const float* p, const float* __restrict x,
                 float* __restrict z, float* __restrict a) const {
        const float* w = p + inicio;
        const float* bias = w + num_pesos();
        for (int i = 0; i < out_features; ++i) {
            z[i] = bias[i];
        }
        // Cada neurônio soma os produtos na ordem das entradas (0, 1, 2...).
        // O laço interno percorre os neurônios: as somas são independentes e
        // o SIMD faz várias de uma vez, sem mudar o resultado.
        for (int j = 0; j < in_features; ++j) {
            const float xj = x[j];
            const float* __restrict wj =
                w + static_cast<size_t>(j) * out_features;
            for (int i = 0; i < out_features; ++i) {
                z[i] += xj * wj[i];
            }
        }
        for (int i = 0; i < out_features; ++i) {
            // As camadas ocultas usam ReLU. A última devolve escores lineares.
            a[i] = final ? z[i] : max(0.0f, z[i]);
        }
    }

    // Calcula o gradiente de uma amostra sem alterar os pesos. grad_in recebe
    // o gradiente da entrada (exceto na primeira camada, onde seria
    // descartado) e g, o gradiente dos parâmetros, no mesmo layout de p.
    void Backward(const float* p, const float* __restrict x,
                  const float* __restrict z, const float* __restrict grad_out,
                  float* __restrict delta, float* __restrict grad_in,
                  float* __restrict g) const {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < out_features; ++i) {
            delta[i] = final ? grad_out[i]
                             : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
        }
        const float* w = p + inicio;
        if (!primeira) {
            for (int j = 0; j < in_features; ++j) {
                const float* wj = w + static_cast<size_t>(j) * out_features;
                float sum = 0;
                for (int i = 0; i < out_features; ++i) {
                    sum += delta[i] * wj[i];
                }
                grad_in[j] = sum;
            }
        }
        // O gradiente do peso é delta vezes a entrada; o do viés, o delta.
        float* gw = g + inicio;
        for (int j = 0; j < in_features; ++j) {
            const float xj = x[j];
            float* __restrict gwj = gw + static_cast<size_t>(j) * out_features;
            for (int i = 0; i < out_features; ++i) {
                gwj[i] = delta[i] * xj;
            }
        }
        float* gb = gw + num_pesos();
        for (int i = 0; i < out_features; ++i) {
            gb[i] = delta[i];
        }
    }
};

struct Rede {
    Camada l1{entradas, ocultos1, false, true, 0};
    Camada l2{ocultos1, ocultos2, false, false, 0};
    Camada l3{ocultos2, classes, true, false, 0};
    vector<float> parametros;

    Rede() {
        l2.inicio = l1.inicio + l1.num_parametros();
        l3.inicio = l2.inicio + l2.num_parametros();
        parametros.assign(l3.inicio + l3.num_parametros(), 0.0f);
        for (const Camada* c : {&l1, &l2, &l3}) {
            inicializar(*c);
        }
    }

    // Semente fixa por camada e mesma ordem de sorteio de sempre (neurônio
    // a neurônio): os pesos iniciais não mudam. O viés começa em zero.
    void inicializar(const Camada& c) {
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        float* w = parametros.data() + c.inicio;
        for (int i = 0; i < c.out_features; ++i) {
            for (int j = 0; j < c.in_features; ++j) {
                w[static_cast<size_t>(j) * c.out_features + i] =
                    distribution(gen);
            }
        }
    }
};

// Buffers de uma amostra. Cada thread tem os seus.
struct Trabalho {
    array<float, ocultos1> z1, a1, d1, ga1;
    array<float, ocultos2> z2, a2, d2, ga2;
    array<float, classes> z3, escores, d3, ge;
};

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const float* escores, int rotulo, float* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            gradiente[i] = erro;
        }
    }
    return total;
}

void inferir(const Rede& r, const float* imagem, Trabalho& t) {
    const float* p = r.parametros.data();
    r.l1.Forward(p, imagem, t.z1.data(), t.a1.data());
    r.l2.Forward(p, t.a1.data(), t.z2.data(), t.a2.data());
    r.l3.Forward(p, t.a2.data(), t.z3.data(), t.escores.data());
}

// Forward, perda e backward de uma amostra. O gradiente dos parâmetros vai
// para g; os pesos não mudam. Devolve a perda da amostra.
float gradiente_amostra(const Rede& r, const float* imagem, int rotulo,
                        Trabalho& t, float* g) {
    const float* p = r.parametros.data();
    inferir(r, imagem, t);
    const float valor = perda(t.escores.data(), rotulo, t.ge.data());
    // Propaga o gradiente da última camada até a primeira.
    r.l3.Backward(p, t.a2.data(), t.z3.data(), t.ge.data(), t.d3.data(),
                  t.ga2.data(), g);
    r.l2.Backward(p, t.a1.data(), t.z2.data(), t.ga2.data(), t.d2.data(),
                  t.ga1.data(), g);
    r.l1.Backward(p, imagem, t.z1.data(), t.ga1.data(), t.d1.data(), nullptr,
                  g);
    return valor;
}

// Lê um parâmetro positivo do ambiente ou usa o valor padrão.
template <typename T>
T parametro(const char* nome, T padrao) {
    const char* valor = getenv(nome);
    if (!valor || !*valor) {
        return padrao;
    }
    istringstream campo(valor);
    T lido{};
    if (!(campo >> lido) || !(campo >> ws).eof() || !(lido > 0)) {
        throw runtime_error(string("Valor invalido em ") + nome + ": " + valor);
    }
    return lido;
}

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }

    vector<Amostra> dados;
    string linha;
    size_t numero_linha = 0;
    while (getline(arquivo, linha)) {
        ++numero_linha;
        istringstream campos(linha);
        Amostra amostra{};
        // Cada atributo deve ser seguido por vírgula, inclusive o último.
        for (int i = 0; i < entradas; ++i) {
            float valor;
            char separador;
            if (!(campos >> valor >> separador) || separador != ',' ||
                !isfinite(valor) || valor < 0 || valor > 16) {
                break;
            }
            // Converte os valores de [0, 16] para [0, 1].
            amostra.imagem.push_back(valor / 16.0f);
        }
        // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
        if (amostra.imagem.size() != entradas ||
            !(campos >> amostra.rotulo) ||
            amostra.rotulo < 0 || amostra.rotulo >= classes ||
            !(campos >> ws).eof()) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
    }
    if (arquivo.bad() || dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

void avaliar(const Rede& rede, const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;
    Trabalho t;
    const auto& escores = t.escores;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        inferir(rede, amostra.imagem.data(), t);
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores.data(), amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        // Mini-batch: os gradientes das amostras de um lote são calculados com
        // os mesmos pesos e aplicados juntos, com passo taxa / tamanho do
        // lote. Com lote 1 e taxa 0.02, é exatamente o SGD por amostra
        // original. Ambos podem ser trocados por BARENN_LOTE e BARENN_TAXA.
        const int lote = Barenn::parametro("BARENN_LOTE", 1);
        const float taxa = Barenn::parametro("BARENN_TAXA", 0.02f);
        // Vai para stderr para não alterar a saída comparada na validação.
        cerr << "configuracao: lote " << lote << " | taxa " << taxa << "\n";

        Barenn::Rede rede;
        const size_t P = rede.parametros.size();
        const int n = static_cast<int>(treino.size());
        // Gradiente de cada amostra do lote, um bloco de P floats por amostra.
        vector<float> grad_lote(static_cast<size_t>(lote) * P);
        vector<float> perda_amostra(n);
        array<double, epocas> perdas{};
        // A atualização percorre os parâmetros em faixas.
        constexpr size_t faixa = 256;
        const size_t faixas = (P + faixa - 1) / faixa;

        // A medição inclui forward, perda, backward e atualização de todas as
        // épocas.
        const auto inicio = chrono::steady_clock::now();
        {
            Barenn::Trabalho t;
            for (int epoca = 0; epoca < epocas; ++epoca) {
                // Uma época percorre todo o conjunto na ordem do arquivo.
                for (int ini = 0; ini < n; ini += lote) {
                    const int b = min(lote, n - ini);
                    // Gradientes das amostras do lote, todos com os mesmos
                    // pesos.
                    for (int k = 0; k < b; ++k) {
                        const auto& amostra = treino[ini + k];
                        perda_amostra[ini + k] = Barenn::gradiente_amostra(
                            rede, amostra.imagem.data(), amostra.rotulo, t,
                            grad_lote.data() + static_cast<size_t>(k) * P);
                    }
                    // Atualização: os gradientes do lote são somados na ordem
                    // das amostras.
                    const float passo = taxa / b;
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
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        Barenn::avaliar(rede, teste);
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
