#!/bin/bash
#SBATCH --job-name=lev_cpu
#SBATCH --output=lev_cpu_%j.out
#SBATCH --partition=pluton_cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --mem=1G

# Roda o binario compilado no head-node (lev_omp), na propria pasta de onde
# o sbatch foi chamado (dentro de ~/scratch). Nao compila nada aqui.
cd "$SLURM_SUBMIT_DIR" || exit 1

echo "== particao, no e pasta de execucao =="
echo "particao: $SLURM_JOB_PARTITION | no: $(hostname)"
pwd

echo "== cabecalhos do sistema no no =="
ls -l /usr/include/features.h 2>&1

echo "== nucleos P e E (hibrido) =="
echo -n "nucleos P (cpu_core): "; cat /sys/devices/cpu_core/cpus 2>/dev/null || echo "nao exposto"
echo -n "nucleos E (cpu_atom): "; cat /sys/devices/cpu_atom/cpus 2>/dev/null || echo "nao exposto"
lscpu --extended=CPU,CORE,MAXMHZ

echo "== cpus que o SLURM liberou para o job =="
echo "nproc: $(nproc)"
taskset -cp $$

echo "== OpenMP com afinidade fixa =="
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
./lev_omp | sort -n -k2
