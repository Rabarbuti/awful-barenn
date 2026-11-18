# Compilacao das versoes do programa.
# No Franky, rode no head-node: os nos de computacao nao tem os cabecalhos
# do sistema. Cada binario ganha um arquivo .info com o compilador, as flags
# e o commit de origem (os nos de computacao nao tem git).

CXX    := g++
STD    := -std=c++17
FONTE  := barenn.cpp
BUILD  := build

COMMIT   := $(shell git rev-parse --short HEAD 2>/dev/null || echo desconhecido)
ALTERADO := $(shell git diff --quiet HEAD -- $(FONTE) Makefile 2>/dev/null || echo " (fonte com alteracoes nao commitadas)")

# Versoes de CPU compiladas a partir do fonte atual.
CPU := $(BUILD)/barenn_O0 $(BUILD)/barenn_O2 $(BUILD)/barenn_O2_pg

FLAGS_barenn_O0    := -O0
FLAGS_barenn_O2    := -O2
FLAGS_barenn_O2_pg := -O2 -pg

.PHONY: cpu baseline clean

# Alvo padrao (make): as versoes de CPU do fonte atual.
cpu: $(CPU)

# Mesmo conjunto; nome mantido para reproduzir a tag v0.0-baseline.
baseline: $(CPU)

$(BUILD)/%: $(FONTE) Makefile | $(BUILD)
	$(CXX) $(STD) $(FLAGS_$*) $(FONTE) -o $@
	@{ $(CXX) --version | head -1; echo "flags: $(STD) $(FLAGS_$*)"; echo "commit: $(COMMIT)$(ALTERADO)"; } > $@.info

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)
