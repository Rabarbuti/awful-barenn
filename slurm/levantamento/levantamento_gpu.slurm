#!/bin/bash
#SBATCH --job-name=lev_gpu2
#SBATCH --output=lev_gpu_%j.out
#SBATCH --partition=pluton_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=2G

# O /scratch so existe nos nos de computacao. O binario e compilado no
# head-node (no $HOME); o job copia para o /scratch do no, roda la e apaga
# a pasta temporaria no fim. A saida (.out) vai para a pasta do sbatch.

TRABALHO="/scratch/$USER/aps2/job_${SLURM_JOB_ID}"
mkdir -p "$TRABALHO" || { echo "ERRO: nao foi possivel criar $TRABALHO"; exit 1; }
trap 'rm -rf "$TRABALHO"' EXIT
cp "$SLURM_SUBMIT_DIR/lev_teste" "$TRABALHO/" || exit 1
cd "$TRABALHO" || exit 1

echo "== particao, no e pasta de execucao =="
echo "particao: $SLURM_JOB_PARTITION | no: $(hostname)"
pwd
df -h /scratch | tail -1

echo "== cabecalhos do sistema no no =="
ls -l /usr/include/features.h 2>&1

module load cuda/12.8.1

echo "== execucao =="
./lev_teste

echo "== Nsight Systems (nsys) =="
nsys profile --stats=true --force-overwrite true -o lev_nsys ./lev_teste 2>&1 | tail -30

echo "== Nsight Compute (ncu) =="
# ERR_NVGPUCTRPERM aqui significa que o ncu nao tem permissao de ler os contadores.
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active ./lev_teste 2>&1 | tail -20
