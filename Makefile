# Compilacao das versoes do programa.
# No Franky, rode no head-node: os nos de computacao nao tem os cabecalhos
# do sistema. Cada binario ganha um arquivo .info com o compilador, as flags
# e o commit de origem (os nos de computacao nao tem git).

CXX    := g++
STD    := -std=c++17
FONTE  := barenn.cpp
COMUM  := barenn.h
BUILD  := build
CUDA_FONTE := barenn_cuda.cu

COMMIT   := $(shell git rev-parse --short HEAD 2>/dev/null || echo desconhecido)
ALTERADO := $(shell git diff --quiet HEAD -- $(FONTE) $(COMUM) $(CUDA_FONTE) Makefile 2>/dev/null || echo " (fonte com alteracoes nao commitadas)")

# Versoes de CPU compiladas a partir do fonte atual.
CPU := $(BUILD)/barenn_O0 $(BUILD)/barenn_O2 $(BUILD)/barenn_O2_pg \
       $(BUILD)/barenn_avx2 $(BUILD)/barenn_avx2_pg $(BUILD)/barenn_omp

# AVX2 dos nos de medicao (Raptor Lake). -ffp-contract=off impede o FMA,
# que mudaria o arredondamento e a saida. Nao usar -march=native: a
# compilacao roda no head-node, que nao e o no de medicao.
AVX2 := -O3 -march=raptorlake -ffp-contract=off

FLAGS_barenn_O0      := -O0
FLAGS_barenn_O2      := -O2
FLAGS_barenn_O2_pg   := -O2 -pg
FLAGS_barenn_avx2    := $(AVX2)
FLAGS_barenn_avx2_pg := $(AVX2) -pg
# AVX2 + OpenMP. Numero de threads: OMP_NUM_THREADS (no SLURM, sbatch -c).
FLAGS_barenn_omp     := $(AVX2) -fopenmp

.PHONY: cpu baseline cuda emulado clean

# Alvo padrao (make): as versoes de CPU do fonte atual.
cpu: $(CPU)

# Mesmo conjunto; nome mantido para reproduzir a tag v0.0-baseline.
baseline: $(CPU)

$(BUILD)/%: $(FONTE) $(COMUM) Makefile | $(BUILD)
	$(CXX) $(STD) $(FLAGS_$*) $(FONTE) -o $@
	@{ $(CXX) --version | head -1; echo "flags: $(STD) $(FLAGS_$*)"; echo "commit: $(COMMIT)$(ALTERADO)"; } > $@.info

$(BUILD):
	mkdir -p $@

# Versao de GPU (RTX A1000 = sm_86). No Franky, antes de compilar:
#   module load cuda/12.8.1
# -fmad=false: sem FMA na GPU, para a saida ser identica a da CPU.
NVCC       := nvcc
CUDA_FLAGS := $(STD) -O3 -arch=sm_86 -fmad=false -Xcompiler "$(AVX2)"

cuda: $(BUILD)/barenn_cuda

$(BUILD)/barenn_cuda: $(CUDA_FONTE) $(COMUM) Makefile | $(BUILD)
	$(NVCC) $(CUDA_FLAGS) $(CUDA_FONTE) -o $@
	@{ $(NVCC) --version | grep release; echo 'flags: $(CUDA_FLAGS)'; echo "commit: $(COMMIT)$(ALTERADO)"; } > $@.info

# Kernels emulados na CPU (g++, sem GPU), so para testar a logica.
emulado: $(BUILD)/barenn_cuda_emulado

$(BUILD)/barenn_cuda_emulado: $(CUDA_FONTE) tests/cuda_emulado.h $(COMUM) Makefile | $(BUILD)
	$(CXX) $(STD) $(AVX2) -x c++ $(CUDA_FONTE) -o $@

clean:
	rm -rf $(BUILD)
