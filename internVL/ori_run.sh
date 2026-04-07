#!/bin/bash

# --- Job Configuration ---
#SBATCH -J internVLJob            # Job name
#SBATCH --partition=gpu           # Partition (queue) name
#SBATCH --gres=gpu:h100-47:1      # Request 1 A100 GPU (40GB)
#SBATCH --ntasks=1
#SBATCH --mem=192G                # System memory
#SBATCH --time=48:00:00           # Time limit (48 hours)

# --- Logging (Critical for debugging) ---
#SBATCH --output=logs/job_%j.out  # Standard output log (%j = job ID)
#SBATCH --error=logs/job_%j.err   # Standard error log

# --- Environment Setup ---
echo "Job started on $(date) on host $(hostname)"

# 1. Create logs directory if it doesn't exist
mkdir -p logs

# 2. Load Modules & Activate Environment (Adjust names as needed)
# module load cuda/11.8 
source ~/.bashrc
conda activate videollama # <--- REPLACE with your actual environment name

# --- Execution ---
echo "Starting evaluation script..."

# srun handles the job step creation
srun python eval_ori.py

echo "Job finished on $(date)"