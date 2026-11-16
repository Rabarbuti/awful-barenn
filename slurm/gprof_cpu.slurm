#!/bin/bash
#SBATCH --job-name=gprof_cpu
#SBATCH --output=resultados/slurm_%x_%j.out
#SBATCH --partition=pluton_cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=2G

# Perfil com gprof: executa uma vez um binario compilado com -pg e gera o
# perfil plano e o grafo de chamadas em resultados/<rotulo>/gprof.txt.
# O perf e o valgrind nao estao instalados nos nos do Franky.
#
# Uso: sbatch slurm/gprof_cpu.slurm <binario_com_-pg> <rotulo>
# Ex.: sbatch slurm/gprof_cpu.slurm build/barenn_O2_pg baseline_O2

BIN="${1:?uso: sbatch slurm/gprof_cpu.slurm <binario_com_-pg> <rotulo>}"
ROTULO="${2:?uso: sbatch slurm/gprof_cpu.slurm <binario_com_-pg> <rotulo>}"

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f optdigits.tra ] && [ -f optdigits.tes ] || { echo "ERRO: rode o sbatch na raiz do repositorio"; exit 1; }
[ -x "$BIN" ] || { echo "ERRO: $BIN nao existe; compile no head-node com make"; exit 1; }

DEST="resultados/$ROTULO"
mkdir -p "$DEST"

# O gmon.out e gravado na pasta atual; roda numa pasta propria do job para
# nao misturar com outra execucao.
TMP="resultados/.gprof_$SLURM_JOB_ID"
mkdir -p "$TMP"
( cd "$TMP" && "../../$BIN" ../../optdigits.tra ../../optdigits.tes > saida.txt ) || { echo "ERRO: execucao falhou"; exit 1; }

{
    echo "binario: $BIN"
    [ -f "$BIN.info" ] && sed 's/^/  /' "$BIN.info"
    echo "particao: $SLURM_JOB_PARTITION | no: $(hostname) | job: $SLURM_JOB_ID"
    echo "cpus do job:$(taskset -cp $$ | cut -d: -f2)"
    echo "aviso: -pg acrescenta instrumentacao; os tempos oficiais vem do bench_cpu.slurm"
    echo
    gprof -b "$BIN" "$TMP/gmon.out"
} > "$DEST/gprof.txt"
mv "$TMP/saida.txt" "$DEST/saida_gprof.txt"
rm -rf "$TMP"

head -40 "$DEST/gprof.txt"
