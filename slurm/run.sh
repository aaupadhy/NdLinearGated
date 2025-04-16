#!/bin/bash
#SBATCH --job-name=NdLinearGated
#SBATCH --output=ablation.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=15:00:00


echo "=== Job started at $(date) ==="
echo "Running on node: $(hostname)"

module purge
module load WebProxy
export http_proxy=http://10.73.132.63:8080
export https_proxy=http://10.73.132.63:8080

PROJECT_DIR="/scratch/user/aaupadhy/college/projects/NdLinearGated/"
cd $PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

COMPUTE_NODE=$(hostname -s)
echo "ssh -N -L 8787:${COMPUTE_NODE}:8787 aaupadhy@grace.hprc.tamu.edu"

source ~/.bashrc
conda activate ML

echo "=== Starting experimentation at $(date) ==="
python experiments/run_ablation.py

echo "=== Python script finished at $(date) with exit code $PYTHON_EXIT_CODE ==="
echo "=== Job finished at $(date) ==="