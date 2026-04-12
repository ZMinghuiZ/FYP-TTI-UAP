#!/bin/bash

#SBATCH -J gemma4_Eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/gemma4_%j.out
#SBATCH --error=logs/gemma4_%j.err

# 1. Environment
source ~/.bashrc
source "${SLURM_SUBMIT_DIR}/config.sh"
conda activate "${CONDA_ENV}"

# 3. Directory
cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "Job started on $(hostname) at $(date)"

# 4. Run
srun python "${REPO_ROOT}/evaluation/gemma4/eval_ori.py" \
    --video_dir "${ADV_OUTPUT_DIR}"

echo "Job finished at $(date)"
