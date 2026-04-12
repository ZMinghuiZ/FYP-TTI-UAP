#!/bin/bash

#SBATCH -J eval_top
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=sweep/logs/eval_top_%A_%a.out
#SBATCH --error=sweep/logs/eval_top_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════
# Evaluate Top Grid Configs on All 4 VLMs (Phase 3) -- PARALLEL
# ══════════════════════════════════════════════════════════════════════════
# Each SLURM array task evaluates ONE config on all 4 VLMs.
# Configs run in parallel on separate GPUs.
#
# Usage:
#   1. Create sweep/eval_top_list.txt with one run name per line:
#        G3_a4-near-s8-low
#        G8_a45-mod-temporal
#        S8_small-mod-temporal
#   2. Set --array to match the number of lines:
#        sbatch --array=1-3 run_sweep_eval_top.sh
#
# Or run a single config locally:
#   bash run_sweep_eval_top.sh G3_a4-near-s8-low
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

source ~/.bashrc
source "${SLURM_SUBMIT_DIR}/config.sh"
conda activate "${CONDA_ENV}"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p sweep/logs

# ── Paths ─────────────────────────────────────────────────────────────────
VIDEO_DIR="${VIDEO_DIR_CLEAN}"

INTERNVL_MODEL="OpenGVLab/InternVL3-38B"
QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
LLAVA_MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
VIDEOLLAMA3_MODEL="DAMO-NLP-SG/VideoLLaMA3-7B"

STRETCH=4
LIST_FILE="sweep/eval_top_list.txt"

# ── Select which config this task evaluates ───────────────────────────────
if [ $# -gt 0 ]; then
    # Local mode: run name passed as argument
    RUN_NAME="$1"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    # Array mode: pick the Nth non-empty, non-comment line from the list
    if [ ! -f "$LIST_FILE" ]; then
        echo "ERROR: ${LIST_FILE} not found."
        exit 1
    fi
    RUN_NAME=$(grep -v '^\s*#' "$LIST_FILE" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p" | xargs)
    if [ -z "$RUN_NAME" ]; then
        echo "ERROR: No config found at line ${SLURM_ARRAY_TASK_ID} in ${LIST_FILE}."
        exit 1
    fi
else
    echo "ERROR: No run name provided."
    echo ""
    echo "Usage:"
    echo "  sbatch --array=1-N run_sweep_eval_top.sh   (reads sweep/eval_top_list.txt)"
    echo "  bash run_sweep_eval_top.sh RUN_NAME         (single config)"
    exit 1
fi

WORK_DIR="$(pwd)"
RUN_DIR="sweep/${RUN_NAME}"
ADV_DIR="${RUN_DIR}/adv_videos"
UAP_PATH="${RUN_DIR}/tti_uap.pt"
EVAL_CWD="${WORK_DIR}/${RUN_DIR}"
ADV_DIR_ABS="${WORK_DIR}/${ADV_DIR}"

echo "=============================================="
echo "Evaluate Config on 4 VLMs"
echo "Run           : ${RUN_NAME}"
echo "Array Task    : ${SLURM_ARRAY_TASK_ID:-local}"
echo "Node          : $(hostname)"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time    : $(date)"
echo "=============================================="
echo ""

if [ ! -f "$UAP_PATH" ]; then
    echo "ERROR: ${UAP_PATH} not found -- skipping."
    exit 1
fi

mkdir -p "$ADV_DIR"

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Apply UAP to videos
# ══════════════════════════════════════════════════════════════════════════
echo ">>> Step 1: Applying UAP to videos ..."

srun python "${REPO_ROOT}/attack/apply_uap.py" \
    --uap "$UAP_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$ADV_DIR" \
    --stretch "$STRETCH" \
    --crf 23

echo "    Adversarial videos saved to ${ADV_DIR}"

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Evaluate on InternVL3
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: Evaluating on InternVL3 ..."

cd "$EVAL_CWD"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'internVL'))
os.chdir('${EVAL_CWD}')
from eval_ori import main
main('${INTERNVL_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"
echo "    InternVL3 done."

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Evaluate on Qwen3-VL
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: Evaluating on Qwen3-VL ..."

cd "$EVAL_CWD"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'qwen3'))
os.chdir('${EVAL_CWD}')
from eval_ori import main
main('${QWEN_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"
echo "    Qwen3-VL done."

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Evaluate on LLaVA-OneVision
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 4: Evaluating on LLaVA-OneVision ..."

cd "$EVAL_CWD"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'llava_onevision'))
os.chdir('${EVAL_CWD}')
from eval_ori import main
main('${LLAVA_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"
echo "    LLaVA-OneVision done."

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Evaluate on VideoLLaMA3
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 5: Evaluating on VideoLLaMA3 ..."

cd "$EVAL_CWD"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'videollama3'))
os.chdir('${EVAL_CWD}')
from eval_ori import main
main('${VIDEOLLAMA3_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"
echo "    VideoLLaMA3 done."

# ══════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo "All 4 VLM evaluations complete for ${RUN_NAME} at $(date)"
echo "=============================================="
