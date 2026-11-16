#!/bin/bash
#SBATCH --job-name=lev_gpu
#SBATCH --output=lev_gpu_%j.out
#SBATCH --partition=pluton_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --mem=2G

# Roda o binario compilado no head-node (lev_teste), na propria pasta de onde
# o sbatch foi chamado (dentro de ~/scratch). Nao compila nada aqui.
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "== particao, no e pasta de execucao =="
echo "particao: $SLURM_JOB_PARTITION | no: $(hostname)"
pwd

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

rm -f lev_nsys.nsys-rep lev_nsys.sqlite
