#!/bin/bash

#SBATCH -J stretch_par
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=sweep/logs/stretch_par_%A_%a.out
#SBATCH --error=sweep/logs/stretch_par_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════
# Parallel Stretch Sweep for a SINGLE Config
# ══════════════════════════════════════════════════════════════════════════
# Each SLURM array task handles one stretch value, so all stretches run
# in parallel on separate GPUs.
#
# Usage:
#   sbatch --array=1-5 run_stretch_parallel.sh S13_a4-no-temporal
#   sbatch --array=1-7 run_stretch_parallel.sh S13_a4-no-temporal  # if 7 stretches
#
# The array index maps to the STRETCHES array below. Adjust STRETCHES and
# the --array range together.
#
# For local testing (single stretch):
#   SLURM_ARRAY_TASK_ID=3 bash run_stretch_parallel.sh S13_a4-no-temporal
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

# ── Stretch values (one per array task) ───────────────────────────────────
STRETCHES=(4 12 24 32 64)
SCALE=1

# ── Parse arguments ───────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "ERROR: No run name provided."
    echo ""
    echo "Usage:"
    echo "  sbatch --array=1-${#STRETCHES[@]} run_stretch_parallel.sh RUN_NAME"
    echo "  SLURM_ARRAY_TASK_ID=1 bash run_stretch_parallel.sh RUN_NAME"
    echo ""
    echo "Stretch values: ${STRETCHES[*]}"
    echo "Array index 1-${#STRETCHES[@]} maps to these values."
    exit 1
fi

RUN_NAME="$1"

# ── Select stretch for this array task ────────────────────────────────────
TASK_ID="${SLURM_ARRAY_TASK_ID:?ERROR: SLURM_ARRAY_TASK_ID not set. Use --array or export it.}"

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "${#STRETCHES[@]}" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [1..${#STRETCHES[@]}]."
    echo "Available stretches: ${STRETCHES[*]}"
    exit 1
fi

STRETCH="${STRETCHES[$((TASK_ID - 1))]}"
COMBO_TAG="stretch${STRETCH}_scale${SCALE}"

WORK_DIR="$(pwd)"
RUN_DIR="sweep/${RUN_NAME}"
UAP_PATH="${RUN_DIR}/tti_uap.pt"
ADV_DIR="${RUN_DIR}/adv_videos_${COMBO_TAG}"
ADV_DIR_ABS="${WORK_DIR}/${ADV_DIR}"

echo "=============================================="
echo "Parallel Stretch Sweep"
echo "Run           : ${RUN_NAME}"
echo "Array Task    : ${TASK_ID}"
echo "Stretch       : ${STRETCH}"
echo "Scale         : ${SCALE}"
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

# ── Apply UAP ─────────────────────────────────────────────────────────────
echo ">>> Applying UAP (stretch=${STRETCH}, scale=${SCALE}) ..."

srun python "${REPO_ROOT}/attack/apply_uap.py" \
    --uap "$UAP_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$ADV_DIR" \
    --stretch "$STRETCH" \
    --scale "$SCALE" \
    --crf 23

echo "    Adversarial videos saved to ${ADV_DIR}"

# ── Evaluate on InternVL3 ─────────────────────────────────────────────────
echo ">>> Evaluating on InternVL3 ..."

cd "$ADV_DIR_ABS"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'internVL'))
os.chdir('${ADV_DIR_ABS}')
from eval_ori import main
main('${INTERNVL_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"
echo "    InternVL3 done."

# ── Evaluate on Qwen3-VL ──────────────────────────────────────────────────
echo ">>> Evaluating on Qwen3-VL ..."

cd "$ADV_DIR_ABS"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'qwen3'))
os.chdir('${ADV_DIR_ABS}')
from eval_ori import main
main('${QWEN_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"
echo "    Qwen3-VL done."

# ── Evaluate on LLaVA-OneVision ────────────────────────────────────────────
echo ">>> Evaluating on LLaVA-OneVision ..."

cd "$ADV_DIR_ABS"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'llava_onevision'))
os.chdir('${ADV_DIR_ABS}')
from eval_ori import main
main('${LLAVA_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"
echo "    LLaVA-OneVision done."

# ── Evaluate on VideoLLaMA3 ────────────────────────────────────────────────
echo ">>> Evaluating on VideoLLaMA3 ..."

cd "$ADV_DIR_ABS"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'videollama3'))
os.chdir('${ADV_DIR_ABS}')
from eval_ori import main
main('${VIDEOLLAMA3_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"
echo "    VideoLLaMA3 done."

echo ""
echo "=============================================="
echo "Stretch ${STRETCH} (scale=${SCALE}) for ${RUN_NAME} finished at $(date)"
echo "=============================================="
