#!/bin/bash

#SBATCH -J ss_par
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=sweep/logs/ss_par_%A_%a.out
#SBATCH --error=sweep/logs/ss_par_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════
# Parallel Stretch × Scale Sweep for a SINGLE Config
# ══════════════════════════════════════════════════════════════════════════
# Each SLURM array task handles one (stretch, scale) pair, so all combos
# run in parallel on separate GPUs.
#
#   Task 1 : stretch=32, scale=0.5   (ε = 8/255)
#   Task 2 : stretch=32, scale=0.625 (ε = 10/255)
#   Task 3 : stretch=32, scale=0.75  (ε = 12/255)
#   Task 4 : stretch=64, scale=0.5   (ε = 8/255)
#   Task 5 : stretch=64, scale=0.625 (ε = 10/255)
#   Task 6 : stretch=64, scale=0.75  (ε = 12/255)
#
# Usage:
#   sbatch --array=1-6 run_stretch_scale_parallel.sh G4_a4-near-s8-high
#   sbatch --array=1-3 run_stretch_scale_parallel.sh G4_a4-near-s8-high  # stretch=32 only
#   sbatch --array=4-6 run_stretch_scale_parallel.sh G4_a4-near-s8-high  # stretch=64 only
#
# For local testing (single combo):
#   SLURM_ARRAY_TASK_ID=1 bash run_stretch_scale_parallel.sh G4_a4-near-s8-high
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

source ~/.bashrc
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
conda activate "${CONDA_ENV}"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p sweep/logs

# ── Paths ─────────────────────────────────────────────────────────────────
VIDEO_DIR="${VIDEO_DIR_CLEAN}"

INTERNVL_MODEL="OpenGVLab/InternVL3-38B"
QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
LLAVA_MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
VIDEOLLAMA3_MODEL="DAMO-NLP-SG/VideoLLaMA3-7B"

# ── (stretch, scale) pairs — one per array task ──────────────────────────
STRETCHES=(32    32    32   64    64    64)
SCALES=(   0.5   0.625 0.75 0.5   0.625 0.75)

N_TASKS="${#STRETCHES[@]}"

# ── Parse arguments ───────────────────────────────────────────────────────
if [ $# -lt 1 ]; then
    echo "ERROR: No run name provided."
    echo ""
    echo "Usage:"
    echo "  sbatch --array=1-${N_TASKS} run_stretch_scale_parallel.sh RUN_NAME"
    echo "  SLURM_ARRAY_TASK_ID=1 bash run_stretch_scale_parallel.sh RUN_NAME"
    echo ""
    echo "Task mapping:"
    for i in $(seq 0 $((N_TASKS - 1))); do
        echo "  $((i + 1)): stretch=${STRETCHES[$i]}, scale=${SCALES[$i]}"
    done
    exit 1
fi

RUN_NAME="$1"

# ── Select (stretch, scale) for this array task ──────────────────────────
TASK_ID="${SLURM_ARRAY_TASK_ID:?ERROR: SLURM_ARRAY_TASK_ID not set. Use --array or export it.}"

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "$N_TASKS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [1..${N_TASKS}]."
    echo "Available combos:"
    for i in $(seq 0 $((N_TASKS - 1))); do
        echo "  $((i + 1)): stretch=${STRETCHES[$i]}, scale=${SCALES[$i]}"
    done
    exit 1
fi

IDX=$((TASK_ID - 1))
STRETCH="${STRETCHES[$IDX]}"
SCALE="${SCALES[$IDX]}"
COMBO_TAG="stretch${STRETCH}_scale${SCALE}"

WORK_DIR="$(pwd)"
RUN_DIR="sweep/${RUN_NAME}"
UAP_PATH="${RUN_DIR}/tti_uap.pt"
ADV_DIR="${RUN_DIR}/adv_videos_${COMBO_TAG}"
ADV_DIR_ABS="${WORK_DIR}/${ADV_DIR}"

echo "=============================================="
echo "Parallel Stretch × Scale Sweep"
echo "=============================================="
echo "  Run           : ${RUN_NAME}"
echo "  Task          : ${TASK_ID}/${N_TASKS}"
echo "  Stretch       : ${STRETCH}"
echo "  Scale         : ${SCALE}"
echo "  Combo tag     : ${COMBO_TAG}"
echo "  Node          : $(hostname)"
echo "  GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start time    : $(date)"
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
echo "Task ${TASK_ID} (stretch=${STRETCH}, scale=${SCALE}) for ${RUN_NAME} finished at $(date)"
echo "=============================================="
