#!/bin/bash
#SBATCH --job-name=lev_cpu2
#SBATCH --output=lev_cpu_%j.out
#SBATCH --partition=pluton_cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --mem=1G

# O /scratch so existe nos nos de computacao. O binario e compilado no
# head-node (no $HOME); o job copia para o /scratch do no, roda la e apaga
# a pasta temporaria no fim. A saida (.out) vai para a pasta do sbatch.

TRABALHO="/scratch/$USER/aps2/job_${SLURM_JOB_ID}"
mkdir -p "$TRABALHO" || { echo "ERRO: nao foi possivel criar $TRABALHO"; exit 1; }
trap 'rm -rf "$TRABALHO"' EXIT
cp "$SLURM_SUBMIT_DIR/lev_omp" "$TRABALHO/" || exit 1
cd "$TRABALHO" || exit 1

echo "== particao, no e pasta de execucao =="
echo "particao: $SLURM_JOB_PARTITION | no: $(hostname)"
pwd
df -h /scratch | tail -1

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
