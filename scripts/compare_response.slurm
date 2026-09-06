#!/bin/bash
#SBATCH --job-name=hfdft-response
#SBATCH --partition=work            # 'work' = 24 cpus, 'large' = 32 cpus
#SBATCH --nodes=1                   # PySCF is OpenMP only -- it CANNOT span nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=hfdft-response-%j.out

# PySCF owns the threads; keep BLAS single-threaded inside them, otherwise
# the two layers oversubscribe the cores (this cost ~2x wall and 32x sys time
# when we measured it).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

source "${HOME}/apps/miniforge3/etc/profile.d/conda.sh"
conda activate pyscf

cd "${SLURM_SUBMIT_DIR}/../tests" || exit 1

echo "host         : $(hostname)"
echo "cpus-per-task: ${SLURM_CPUS_PER_TASK:-?}"
echo

# Push until it gets slow -- the point is to find where the Z-vector starts
# winning, which needs bigger systems than a laptop can reach.
# Args: <max number of waters> <basis>
time python compare_response.py 8 6-31g
