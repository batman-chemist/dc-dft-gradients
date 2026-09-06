#!/bin/bash
#SBATCH --job-name=hfdft-validate
#SBATCH --partition=work            # 'work' = 24 cpus, 'large' = 32 cpus
#SBATCH --nodes=1                   # PySCF is OpenMP only -- it CANNOT span nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=hfdft-validate-%j.out

# PySCF parallelism is shared-memory (OpenMP) plus threaded BLAS.
# There is no MPI here: asking for >1 node would leave the extra nodes idle.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

# Keep BLAS from fighting PySCF's own threads over the same cores.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

source "${HOME}/apps/miniforge3/etc/profile.d/conda.sh"
conda activate pyscf

cd "${SLURM_SUBMIT_DIR}/../tests" || exit 1

echo "host        : $(hostname)"
echo "cpus-per-task: ${SLURM_CPUS_PER_TASK:-?}"
python -c "from pyscf import lib; print('pyscf threads:', lib.num_threads())"
echo

time python validate.py
