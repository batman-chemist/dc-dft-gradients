#!/bin/bash
#SBATCH --job-name=dcdft-response
#SBATCH --nodes=1                   # PySCF is OpenMP only -- it CANNOT span nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=dcdft-response-%j.out
#
# Submit with your site's partition/account, e.g.
#     sbatch --partition=<partition> compare_response.slurm
#
# Cross-checks the Z-vector response term against the per-atom CPHF reference
# and benchmarks both across system sizes, to find where the Z-vector starts
# paying off on real hardware.

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
echo

# Args: <max number of waters> <basis>.  Raise the first until it gets slow --
# the interesting behaviour is at sizes a laptop cannot reach.
time python compare_response.py 8 6-31g
