#!/bin/bash

#SBATCH -J qwenVL_Eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/qwen_%j.out
#SBATCH --error=logs/qwen_%j.err

# 1. Environment
source ~/.bashrc
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/config.sh"
conda activate "${CONDA_ENV}"

# 3. Directory
cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Job started on $(hostname) at $(date)"

# 4. Run
srun python "${REPO_ROOT}/evaluation/qwen3/eval_ori.py" \
    --video_dir "${ADV_OUTPUT_DIR}"

echo "Job finished at $(date)"