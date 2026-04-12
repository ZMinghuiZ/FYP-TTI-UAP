#!/bin/bash

#SBATCH -J sweep_ss
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=sweep/logs/sweep_ss_%A_%a.out
#SBATCH --error=sweep/logs/sweep_ss_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════
# Stretch × Scale Sweep -- Apply & Evaluate on 4 VLMs
# ══════════════════════════════════════════════════════════════════════════
# For each UAP config in eval_top_list.txt, applies the UAP with every
# (stretch, scale) combination and evaluates on all 4 VLMs.
#
# Grid:  stretch ∈ {14, 16, 18}  ×  scale ∈ {1, 0.75, 0.5}  →  9 combos
# Output: sweep/<RUN_NAME>/adv_videos_stretch<S>_scale<X>/
#
# Usage:
#   sbatch --array=1-N run_sweep_stretch_scale.sh
#   bash run_sweep_stretch_scale.sh RUN_NAME
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

LIST_FILE="sweep/eval_top_list.txt"

# ── Sweep grid ────────────────────────────────────────────────────────────
STRETCHES=(14 16 18)
SCALES=(1 0.75 0.5)

# ── Optional post-processing (set to 0 to skip) ──────────────────────────
NOISE_STD=0
BILATERAL_D=0
BILATERAL_SIGMA_COLOR=75
BILATERAL_SIGMA_SPACE=75
PP_SEED=42

# ── Select which config this task evaluates ───────────────────────────────
if [ $# -gt 0 ]; then
    RUN_NAME="$1"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
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
    echo "  sbatch --array=1-N run_sweep_stretch_scale.sh   (reads ${LIST_FILE})"
    echo "  bash run_sweep_stretch_scale.sh RUN_NAME         (single config)"
    exit 1
fi

WORK_DIR="$(pwd)"
RUN_DIR="sweep/${RUN_NAME}"
UAP_PATH="${RUN_DIR}/tti_uap.pt"

echo "=============================================="
echo "Stretch × Scale Sweep"
echo "Run           : ${RUN_NAME}"
echo "Array Task    : ${SLURM_ARRAY_TASK_ID:-local}"
echo "Node          : $(hostname)"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time    : $(date)"
echo "Stretches     : ${STRETCHES[*]}"
echo "Scales        : ${SCALES[*]}"
echo "Combos        : $(( ${#STRETCHES[@]} * ${#SCALES[@]} ))"
echo "=============================================="
echo ""

if [ ! -f "$UAP_PATH" ]; then
    echo "ERROR: ${UAP_PATH} not found -- skipping."
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# Loop over all (stretch, scale) combinations
# ══════════════════════════════════════════════════════════════════════════
COMBO=0
TOTAL=$(( ${#STRETCHES[@]} * ${#SCALES[@]} ))

for STRETCH in "${STRETCHES[@]}"; do
    for SCALE in "${SCALES[@]}"; do
        COMBO=$(( COMBO + 1 ))
        COMBO_TAG="stretch${STRETCH}_scale${SCALE}"
        ADV_DIR="${RUN_DIR}/adv_videos_${COMBO_TAG}"
        ADV_DIR_ABS="${WORK_DIR}/${ADV_DIR}"

        mkdir -p "$ADV_DIR"

        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  Combo ${COMBO}/${TOTAL}: ${COMBO_TAG}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        # ── Apply UAP ─────────────────────────────────────────────────
        echo ">>> Applying UAP (stretch=${STRETCH}, scale=${SCALE}) ..."

        srun python "${REPO_ROOT}/attack/apply_uap.py" \
            --uap "$UAP_PATH" \
            --video_dir "$VIDEO_DIR" \
            --output_dir "$ADV_DIR" \
            --stretch "$STRETCH" \
            --scale "$SCALE" \
            --crf 23

        echo "    Adversarial videos saved to ${ADV_DIR}"

        # ── Optional post-processing ──────────────────────────────────
        EVAL_DIR="$ADV_DIR"
        EVAL_DIR_ABS="$ADV_DIR_ABS"
        if [ "$NOISE_STD" != "0" ] || [ "$BILATERAL_D" != "0" ]; then
            PP_SUFFIX="_n${NOISE_STD}_b${BILATERAL_D}"
            EVAL_DIR="${ADV_DIR}${PP_SUFFIX}"
            EVAL_DIR_ABS="${WORK_DIR}/${EVAL_DIR}"
            mkdir -p "$EVAL_DIR"

            echo ">>> Post-processing (noise_std=${NOISE_STD}, bilateral_d=${BILATERAL_D}) ..."
            srun python "${REPO_ROOT}/attack/postprocess_videos.py" \
                --video_dir "$ADV_DIR" \
                --output_dir "$EVAL_DIR" \
                --noise_std "$NOISE_STD" \
                --bilateral_d "$BILATERAL_D" \
                --bilateral_sigma_color "$BILATERAL_SIGMA_COLOR" \
                --bilateral_sigma_space "$BILATERAL_SIGMA_SPACE" \
                --crf 23 \
                --seed "$PP_SEED"
            echo "    Post-processed videos saved to ${EVAL_DIR}"
        fi

        # ── Evaluate on InternVL3 ────────────────────────────────────
        echo ">>> Evaluating on InternVL3 ..."

        cd "$EVAL_DIR_ABS"
        srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'internVL'))
os.chdir('${EVAL_DIR_ABS}')
from eval_ori import main
main('${INTERNVL_MODEL}', '${EVAL_DIR_ABS}')
"
        cd "$WORK_DIR"
        echo "    InternVL3 done."

        # ── Evaluate on Qwen3-VL ─────────────────────────────────────
        echo ">>> Evaluating on Qwen3-VL ..."

        cd "$EVAL_DIR_ABS"
        srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'qwen3'))
os.chdir('${EVAL_DIR_ABS}')
from eval_ori import main
main('${QWEN_MODEL}', '${EVAL_DIR_ABS}')
"
        cd "$WORK_DIR"
        echo "    Qwen3-VL done."

        # ── Evaluate on LLaVA-OneVision ───────────────────────────────
        echo ">>> Evaluating on LLaVA-OneVision ..."

        cd "$EVAL_DIR_ABS"
        srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'llava_onevision'))
os.chdir('${EVAL_DIR_ABS}')
from eval_ori import main
main('${LLAVA_MODEL}', '${EVAL_DIR_ABS}')
"
        cd "$WORK_DIR"
        echo "    LLaVA-OneVision done."

        # ── Evaluate on VideoLLaMA3 ───────────────────────────────────
        echo ">>> Evaluating on VideoLLaMA3 ..."

        cd "$EVAL_DIR_ABS"
        srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'videollama3'))
os.chdir('${EVAL_DIR_ABS}')
from eval_ori import main
main('${VIDEOLLAMA3_MODEL}', '${EVAL_DIR_ABS}')
"
        cd "$WORK_DIR"
        echo "    VideoLLaMA3 done."

        echo ""
        echo "  Combo ${COMBO}/${TOTAL} complete: ${COMBO_TAG}"
        echo ""
    done
done

echo "=============================================="
echo "All ${TOTAL} stretch×scale combos for ${RUN_NAME} finished at $(date)"
echo "=============================================="
