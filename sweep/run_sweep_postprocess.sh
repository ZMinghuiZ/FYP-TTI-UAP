#!/bin/bash

#SBATCH -J sweep_pp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=sweep/logs/sweep_pp_%A_%a.out
#SBATCH --error=sweep/logs/sweep_pp_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════
# Post-process Sweep -- Noise + Bilateral Filter + 4-VLM Eval
# ══════════════════════════════════════════════════════════════════════════
# Each SLURM array task takes one existing adv_videos directory, applies
# Gaussian noise + bilateral filtering, then evaluates on all 4 VLMs.
#
# Setup:
#   1. Create sweep/postprocess_list.txt with one directory per line:
#        sweep/G3_a4-near-s8-low/adv_videos_stretch14_scale1
#        sweep/G3_a4-near-s8-low/adv_videos_stretch16_scale0.75
#   2. Configure NOISE_STD / BILATERAL_D below.
#   3. Submit:
#        sbatch --array=1-N run_sweep_postprocess.sh
#
# Or run a single directory locally:
#   bash run_sweep_postprocess.sh sweep/G3_.../adv_videos_stretch14_scale1
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

source ~/.bashrc
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
conda activate "${CONDA_ENV}"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p sweep/logs

# ── Paths ─────────────────────────────────────────────────────────────────
INTERNVL_MODEL="OpenGVLab/InternVL3-38B"
QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
LLAVA_MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
VIDEOLLAMA3_MODEL="DAMO-NLP-SG/VideoLLaMA3-7B"

LIST_FILE="sweep/postprocess_list.txt"

# ── Post-processing configurations ────────────────────────────────────────
# Format: CONFIG_NAME  NOISE_STD  BILATERAL_D  SIGMA_COLOR  SIGMA_SPACE
declare -A PP_CONFIGS
PP_CONFIGS[1]="mild         3   5   50   50"
PP_CONFIGS[2]="moderate     5   7   75   75"
PP_CONFIGS[3]="heavy        10  9   100  100"
PP_CONFIGS[4]="noise_only   5   0   0    0"
PP_CONFIGS[5]="filter_only  0   7   75   75"

CRF=23
SEED=42

# ── Select which directory this task processes ────────────────────────────
if [ $# -gt 0 ]; then
    SRC_DIR="$1"
elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    if [ ! -f "$LIST_FILE" ]; then
        echo "ERROR: ${LIST_FILE} not found."
        exit 1
    fi
    SRC_DIR=$(grep -v '^\s*#' "$LIST_FILE" | grep -v '^\s*$' | sed -n "${SLURM_ARRAY_TASK_ID}p" | xargs)
    if [ -z "$SRC_DIR" ]; then
        echo "ERROR: No entry at line ${SLURM_ARRAY_TASK_ID} in ${LIST_FILE}."
        exit 1
    fi
else
    echo "ERROR: No source directory provided."
    echo ""
    echo "Usage:"
    echo "  sbatch --array=1-N run_sweep_postprocess.sh   (reads ${LIST_FILE})"
    echo "  bash run_sweep_postprocess.sh <adv_videos_dir>"
    exit 1
fi

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: Directory not found: ${SRC_DIR}"
    exit 1
fi

WORK_DIR="$(pwd)"
TOTAL=${#PP_CONFIGS[@]}

echo "=============================================="
echo "Post-process + Evaluate"
echo "Source        : ${SRC_DIR}"
echo "Array Task    : ${SLURM_ARRAY_TASK_ID:-local}"
echo "Node          : $(hostname)"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time    : $(date)"
echo "Configs       : ${TOTAL}"
echo "=============================================="
echo ""

# ══════════════════════════════════════════════════════════════════════════
# Loop over all post-processing configurations
# ══════════════════════════════════════════════════════════════════════════
CFG_IDX=0

for KEY in $(echo "${!PP_CONFIGS[@]}" | tr ' ' '\n' | sort -n); do
    read -r CFG_NAME NOISE_STD BILATERAL_D SIGMA_COLOR SIGMA_SPACE \
        <<< "${PP_CONFIGS[$KEY]}"
    CFG_IDX=$(( CFG_IDX + 1 ))

    PP_SUFFIX="_${CFG_NAME}"
    OUT_DIR="${SRC_DIR}${PP_SUFFIX}"
    OUT_DIR_ABS="${WORK_DIR}/${OUT_DIR}"

    mkdir -p "$OUT_DIR"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Config ${CFG_IDX}/${TOTAL}: ${CFG_NAME}"
    echo "  noise_std=${NOISE_STD}  bilateral_d=${BILATERAL_D}" \
         " sigma_color=${SIGMA_COLOR}  sigma_space=${SIGMA_SPACE}"
    echo "  output: ${OUT_DIR}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Post-process videos ───────────────────────────────────────────
    echo ">>> Post-processing videos ..."

    PP_ARGS=(
        --video_dir "$SRC_DIR"
        --output_dir "$OUT_DIR"
        --crf "$CRF"
        --seed "$SEED"
    )
    if [ "$NOISE_STD" != "0" ]; then
        PP_ARGS+=(--noise_std "$NOISE_STD")
    fi
    if [ "$BILATERAL_D" != "0" ]; then
        PP_ARGS+=(--bilateral_d "$BILATERAL_D"
                  --bilateral_sigma_color "$SIGMA_COLOR"
                  --bilateral_sigma_space "$SIGMA_SPACE")
    fi

    srun python "${REPO_ROOT}/attack/postprocess_videos.py" "${PP_ARGS[@]}"

    echo "    Post-processed videos saved to ${OUT_DIR}"

    # ── Evaluate on InternVL3 ─────────────────────────────────────────
    echo ">>> Evaluating on InternVL3 ..."

    cd "$OUT_DIR_ABS"
    srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'internVL'))
os.chdir('${OUT_DIR_ABS}')
from eval_ori import main
main('${INTERNVL_MODEL}', '${OUT_DIR_ABS}')
"
    cd "$WORK_DIR"
    echo "    InternVL3 done."

    # ── Evaluate on Qwen3-VL ──────────────────────────────────────────
    echo ">>> Evaluating on Qwen3-VL ..."

    cd "$OUT_DIR_ABS"
    srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'qwen3'))
os.chdir('${OUT_DIR_ABS}')
from eval_ori import main
main('${QWEN_MODEL}', '${OUT_DIR_ABS}')
"
    cd "$WORK_DIR"
    echo "    Qwen3-VL done."

    # ── Evaluate on LLaVA-OneVision ───────────────────────────────────
    echo ">>> Evaluating on LLaVA-OneVision ..."

    cd "$OUT_DIR_ABS"
    srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'llava_onevision'))
os.chdir('${OUT_DIR_ABS}')
from eval_ori import main
main('${LLAVA_MODEL}', '${OUT_DIR_ABS}')
"
    cd "$WORK_DIR"
    echo "    LLaVA-OneVision done."

    # ── Evaluate on VideoLLaMA3 ───────────────────────────────────────
    echo ">>> Evaluating on VideoLLaMA3 ..."

    cd "$OUT_DIR_ABS"
    srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'videollama3'))
os.chdir('${OUT_DIR_ABS}')
from eval_ori import main
main('${VIDEOLLAMA3_MODEL}', '${OUT_DIR_ABS}')
"
    cd "$WORK_DIR"
    echo "    VideoLLaMA3 done."

    echo ""
    echo "  Config ${CFG_IDX}/${TOTAL} complete: ${CFG_NAME}"
    echo ""
done

echo "=============================================="
echo "All ${TOTAL} configs complete for ${SRC_DIR} at $(date)"
echo "=============================================="
