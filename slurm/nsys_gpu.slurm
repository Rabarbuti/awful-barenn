#!/bin/bash
#SBATCH --job-name=nsys_gpu
#SBATCH --output=resultados/slurm_%x_%j.out
#SBATCH --partition=pluton_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --mem=4G

# Perfil com Nsight Systems: tempo dos kernels, das copias entre CPU e GPU e
# das chamadas da API CUDA. O resumo vai para resultados/<rotulo>/nsys.txt e
# o relatorio completo para resultados/<rotulo>/nsys.nsys-rep (abre na
# interface grafica do Nsight Systems). Rode o sbatch na raiz do repositorio,
# depois de compilar no head-node (module load cuda/12.8.1 && make cuda).
#
# Uso: sbatch slurm/nsys_gpu.slurm <binario> <rotulo>
# Ex.: sbatch slurm/nsys_gpu.slurm build/barenn_cuda bloco5_cuda

BIN="${1:?uso: sbatch slurm/nsys_gpu.slurm <binario> <rotulo>}"
ROTULO="${2:?uso: sbatch slurm/nsys_gpu.slurm <binario> <rotulo>}"

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f optdigits.tra ] && [ -f optdigits.tes ] || { echo "ERRO: rode o sbatch na raiz do repositorio"; exit 1; }
[ -x "$BIN" ] || { echo "ERRO: $BIN nao existe; compile no head-node com make cuda"; exit 1; }

DEST="resultados/$ROTULO"
mkdir -p "$DEST"
module load cuda/12.8.1

{
    echo "binario: $BIN"
    [ -f "$BIN.info" ] && sed 's/^/  /' "$BIN.info"
    echo "particao: $SLURM_JOB_PARTITION | no: $(hostname) | job: $SLURM_JOB_ID"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/gpu: /'
    echo "BARENN_LOTE=${BARENN_LOTE:-padrao} BARENN_TAXA=${BARENN_TAXA:-padrao}"
    nsys --version
    echo
    nsys profile --stats=true --force-overwrite true -o "$DEST/nsys" "$BIN" 2>&1
} > "$DEST/nsys.txt"
rm -f "$DEST/nsys.sqlite"

grep -A14 -E "'(cuda_api_sum|cuda_gpu_kern_sum|cuda_gpu_mem_time_sum)'" "$DEST/nsys.txt"
