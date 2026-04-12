#!/bin/bash

#SBATCH -J exp_static
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=sweep/logs/exp_static_%j.out
#SBATCH --error=sweep/logs/exp_static_%j.err

# ══════════════════════════════════════════════════════════════════════════
# Experiment: Static UAP Baseline
# ══════════════════════════════════════════════════════════════════════════
# Applies a pre-made static UAP image to every frame of each eval video
# via alpha blending, then evaluates on all 4 VLMs.
#
# No training step — the UAP is a fixed image (static/000000.jpg).
#
# Usage:
#   sbatch run_exp_static.sh
#   bash run_exp_static.sh
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

source ~/.bashrc
source "${SLURM_SUBMIT_DIR}/config.sh"
conda activate "${CONDA_ENV}"

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p sweep/logs

# ── Paths ─────────────────────────────────────────────────────────────────
VIDEO_DIR="${VIDEO_DIR_CLEAN}"
STATIC_PATCH="${REPO_ROOT}/static/000000.jpg"

# ── Experiment config ─────────────────────────────────────────────────────
RUN_NAME="EXP_static"
RUN_DIR="sweep/${RUN_NAME}"
ALPHA=$(python3 -c "print(16/255)")
CRF=23

ADV_DIR="${RUN_DIR}/adv_videos"

# ── VLM models for evaluation ────────────────────────────────────────────
INTERNVL_MODEL="OpenGVLab/InternVL3-38B"
QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
LLAVA_MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
VIDEOLLAMA3_MODEL="DAMO-NLP-SG/VideoLLaMA3-7B"

WORK_DIR="$(pwd)"
mkdir -p "$RUN_DIR"

echo "=============================================="
echo "Experiment: Static UAP Baseline"
echo "=============================================="
echo "  Run name      : ${RUN_NAME}"
echo "  Node          : $(hostname)"
echo "  GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start time    : $(date)"
echo "----------------------------------------------"
echo "  Static patch  : ${STATIC_PATCH}"
echo "  Alpha         : ${ALPHA}  (16/255)"
echo "  CRF           : ${CRF}"
echo "  Video dir     : ${VIDEO_DIR}"
echo "=============================================="
echo ""

# Save config metadata (compatible with summarize.py)
cat > "${RUN_DIR}/grid_config.txt" <<CFGEOF
alpha_blend=${ALPHA}
crf=${CRF}
patch=${STATIC_PATCH}
experiment=exp_static
CFGEOF

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Apply static UAP to videos
# ══════════════════════════════════════════════════════════════════════════
echo ">>> Step 1: Applying static UAP to videos ..."

mkdir -p "$ADV_DIR"

srun python "${REPO_ROOT}/attack/apply_static_uap.py" \
    --patch "$STATIC_PATCH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$ADV_DIR" \
    --alpha "$ALPHA" \
    --crf "$CRF"

echo "    Adversarial videos saved to ${ADV_DIR}"
echo ""

ADV_DIR_ABS="${WORK_DIR}/${ADV_DIR}"

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Evaluate on all 4 VLMs
# ══════════════════════════════════════════════════════════════════════════

# ── Evaluate on InternVL3 ──────────────────────────────────────────
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

# ── Evaluate on Qwen3-VL ──────────────────────────────────────────
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

# ── Evaluate on LLaVA-OneVision ───────────────────────────────────
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

# ── Evaluate on VideoLLaMA3 ───────────────────────────────────────
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
echo "Experiment (Static UAP) finished at $(date)"
echo "Results in: ${RUN_DIR}/"
echo "=============================================="
