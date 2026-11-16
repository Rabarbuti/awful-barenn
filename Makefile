# Compilacao das versoes do programa.
# No Franky, rode no head-node: os nos de computacao nao tem os cabecalhos
# do sistema. Cada binario ganha um arquivo .info com o compilador e as flags.

CXX    := g++
STD    := -std=c++17
FONTE  := barenn.cpp
BUILD  := build

# Baseline do codigo legado, sem nenhuma alteracao no fonte.
BASELINE := $(BUILD)/barenn_O0 $(BUILD)/barenn_O2 $(BUILD)/barenn_O2_pg

FLAGS_barenn_O0    := -O0
FLAGS_barenn_O2    := -O2
FLAGS_barenn_O2_pg := -O2 -pg

.PHONY: baseline clean

baseline: $(BASELINE)

$(BUILD)/%: $(FONTE) Makefile | $(BUILD)
	$(CXX) $(STD) $(FLAGS_$*) $(FONTE) -o $@
	@{ $(CXX) --version | head -1; echo "flags: $(STD) $(FLAGS_$*)"; } > $@.info

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)
