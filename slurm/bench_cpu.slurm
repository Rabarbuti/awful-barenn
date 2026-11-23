#!/bin/bash
#SBATCH --job-name=bench_cpu
#SBATCH --output=resultados/slurm_%x_%j.out
#SBATCH --partition=pluton_cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=2G

# Executa um binario N vezes e guarda saidas, tempos, estatisticas e ambiente
# em resultados/<rotulo>/. Rode o sbatch na raiz do repositorio, depois de
# compilar no head-node (make).
#
# Uso: sbatch slurm/bench_cpu.slurm <binario> <rotulo> [repeticoes]
# Ex.: sbatch slurm/bench_cpu.slurm build/barenn_O2 baseline_O2 10
# Para mais threads: sbatch -c 8 slurm/bench_cpu.slurm ...
# Outra saida de referencia: REFERENCIA=ref/outra.txt sbatch ...
# (o sbatch repassa as variaveis de ambiente, como BARENN_LOTE e BARENN_TAXA)

BIN="${1:?uso: sbatch slurm/bench_cpu.slurm <binario> <rotulo> [repeticoes]}"
ROTULO="${2:?uso: sbatch slurm/bench_cpu.slurm <binario> <rotulo> [repeticoes]}"
REP="${3:-10}"
REF="${REFERENCIA:-ref/saida_referencia.txt}"

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f optdigits.tra ] && [ -f optdigits.tes ] || { echo "ERRO: rode o sbatch na raiz do repositorio"; exit 1; }
[ -x "$BIN" ] || { echo "ERRO: $BIN nao existe; compile no head-node com make"; exit 1; }

DEST="resultados/$ROTULO"
mkdir -p "$DEST"
rm -f "$DEST"/saida_[0-9]*.txt "$DEST/tempos.txt"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"

# Ambiente da medicao (vai para a Issue).
{
    echo "data: $(date -Iseconds)"
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        estado=$(git diff --quiet HEAD -- . ':!resultados' 2>/dev/null && echo "limpo" || echo "COM ALTERACOES NAO COMMITADAS")
        echo "commit: $(git rev-parse --short HEAD) ($(git branch --show-current), $estado)"
    fi
    echo "binario: $BIN"
    [ -f "$BIN.info" ] && sed 's/^/  /' "$BIN.info"
    echo "comando: $BIN (entradas padrao optdigits.tra e optdigits.tes)"
    echo "repeticoes: $REP"
    echo "particao: $SLURM_JOB_PARTITION | no: $(hostname) | job: $SLURM_JOB_ID"
    echo "cpus do job:$(taskset -cp $$ | cut -d: -f2)"
    echo "OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PLACES=$OMP_PLACES OMP_PROC_BIND=$OMP_PROC_BIND"
    echo "BARENN_LOTE=${BARENN_LOTE:-padrao} BARENN_TAXA=${BARENN_TAXA:-padrao} | referencia: $REF"
    echo "jobs no mesmo no (inclui este): $(squeue -h -w "$(hostname)" 2>/dev/null | wc -l)"
    lscpu | grep -E "Model name|^CPU\(s\)|L1d|L2|L3"
} > "$DEST/ambiente.txt"
cat "$DEST/ambiente.txt"

# Execucoes. "treino" e o tempo que o proprio programa mede (so o laco de
# treino); "total" e o tempo do processo inteiro, incluindo leitura e avaliacao.
echo "# execucao treino_s total_s" > "$DEST/tempos.txt"
for i in $(seq 1 "$REP"); do
    t0=$(date +%s.%N)
    "$BIN" > "$DEST/saida_$i.txt" 2> "$DEST/configuracao.txt"
    status=$?
    t1=$(date +%s.%N)
    if [ $status -ne 0 ]; then
        echo "ERRO: execucao $i terminou com codigo $status"
        exit 1
    fi
    treino=$(awk '/^Tempo de treino/ {print $4}' "$DEST/saida_$i.txt")
    total=$(awk -v a="$t0" -v b="$t1" 'BEGIN {printf "%.4f", b - a}')
    echo "$i $treino $total" >> "$DEST/tempos.txt"
done

estatisticas() {
    sort -g | awk '{v[NR] = $1; s += $1; q += $1 * $1}
        END {
            n = NR; m = s / n
            med = (n % 2) ? v[(n + 1) / 2] : (v[n / 2] + v[n / 2 + 1]) / 2
            var = (n > 1) ? (q - n * m * m) / (n - 1) : 0
            dp = (var > 0) ? sqrt(var) : 0
            printf "n=%d mediana=%.4f min=%.4f max=%.4f media=%.4f desvio=%.4f\n", n, med, v[1], v[n], m, dp
        }'
}

# Corretude: todas as execucoes devem dar a mesma saida e, se ja existir
# referencia, a mesma saida da referencia.
{
    echo "rotulo: $ROTULO"
    echo "treino_s: $(grep -v '^#' "$DEST/tempos.txt" | awk '{print $2}' | estatisticas)"
    echo "total_s:  $(grep -v '^#' "$DEST/tempos.txt" | awk '{print $3}' | estatisticas)"
    deterministico=sim
    for i in $(seq 2 "$REP"); do
        scripts/validar.sh "$DEST/saida_$i.txt" "$DEST/saida_1.txt" > /dev/null || deterministico=NAO
    done
    echo "execucoes com saida identica entre si: $deterministico"
    grep -m1 "^configuracao:" "$DEST/configuracao.txt" 2>/dev/null
    if [ -f "$REF" ]; then
        echo "validacao contra a referencia: $(scripts/validar.sh "$DEST/saida_1.txt" "$REF" | tail -1)"
    else
        echo "validacao contra a referencia: $REF ainda nao existe"
    fi
} > "$DEST/resumo.txt"
cat "$DEST/resumo.txt"
