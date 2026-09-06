#!/bin/bash
#SBATCH --job-name=dcdft-validate
#SBATCH --nodes=1                   # PySCF is OpenMP only -- it CANNOT span nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=dcdft-validate-%j.out
#
# Submit with your site's partition/account, e.g.
#     sbatch --partition=<partition> validate.slurm
#
# Runs the finite-difference validation suite (~a few minutes).

# PySCF owns the threads; keep BLAS single-threaded inside them. Letting both
# layers thread oversubscribes the cores -- measured at 2x wall time and 32x
# system time on a 24-core node, for identical results.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Site-specific setup (conda/module/venv activation). Not tracked in git --
# copy env.sh.example to env.sh and edit for your machine.
if [ -f "${SLURM_SUBMIT_DIR}/env.sh" ]; then
    source "${SLURM_SUBMIT_DIR}/env.sh"
fi

cd "${SLURM_SUBMIT_DIR}/../tests" || exit 1

echo "host          : $(hostname)"
echo "cpus-per-task : ${SLURM_CPUS_PER_TASK:-?}"
python -c "from pyscf import lib; print('pyscf threads :', lib.num_threads())"
echo

time python validate.py
