#!/bin/bash

#SBATCH -J temporal_abl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=sweep/logs/temporal_abl_%A_%a.out
#SBATCH --error=sweep/logs/temporal_abl_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════
# Experiment: Temporal Order Ablation  (parallel via SLURM array)
# ══════════════════════════════════════════════════════════════════════════
# Parameterized via environment variables (defaults to G4 at stretch=12):
#
#   UAP_PATH   – path to trained btc_uap.pt   (default: G4)
#   RUN_TAG    – prefix for output dirs        (default: empty = legacy naming)
#   STRETCHES  – space-separated stretch list  (default: "12")
#
# Task matrix = 5 ablation variants × len(STRETCHES).
# Variants (per stretch):
#   +0 : shuffle seed 42
#   +1 : shuffle seed 123
#   +2 : shuffle seed 999
#   +3 : static frame 0
#   +4 : reversed frame order
#
# Usage:
#   # G4 at stretch=12 (backward compatible, 5 tasks)
#   sbatch --array=1-5 run_exp_temporal_ablation.sh
#
#   # S13 at stretch=12 and stretch=24 (10 tasks)
#   UAP_PATH=sweep/S13_no-temporal-small/btc_uap.pt RUN_TAG=S13 STRETCHES="12 24" \
#     sbatch --array=1-10 run_exp_temporal_ablation.sh
#
#   # Any config at a single stretch
#   UAP_PATH=sweep/G7_some-config/btc_uap.pt RUN_TAG=G7 STRETCHES="12" \
#     sbatch --array=1-5 run_exp_temporal_ablation.sh
#
#   # Local test
#   UAP_PATH=sweep/G4_a4-near-s8-high/btc_uap.pt SLURM_ARRAY_TASK_ID=1 \
#     bash run_exp_temporal_ablation.sh
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

source ~/.bashrc
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
conda activate "${CONDA_ENV}"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p sweep/logs

# ── Configurable paths (override via environment) ─────────────────────────
VIDEO_DIR="${VIDEO_DIR_CLEAN}"
UAP_PATH="${UAP_PATH:-sweep/G4_a4-near-s8-high/btc_uap.pt}"
RUN_TAG="${RUN_TAG:-}"
CRF=23

read -ra STRETCH_ARR <<< "${STRETCHES:-12}"
N_STRETCHES="${#STRETCH_ARR[@]}"

# ── VLM models ────────────────────────────────────────────────────────────
INTERNVL_MODEL="OpenGVLab/InternVL3-38B"
QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
LLAVA_MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
VIDEOLLAMA3_MODEL="DAMO-NLP-SG/VideoLLaMA3-7B"

# ── Ablation variants (5 per stretch) ─────────────────────────────────────
VARIANT_NAMES=( "shuffle_s42" "shuffle_s123" "shuffle_s999" "static_frame0" "reversed" )
APPLY_ARGS=(
    "--shuffle_seed 42"
    "--shuffle_seed 123"
    "--shuffle_seed 999"
    "--static_frame 0"
    "--reverse"
)
N_VARIANTS="${#VARIANT_NAMES[@]}"
N_TASKS=$((N_VARIANTS * N_STRETCHES))

# ── Select variant and stretch for this array task ─────────────────────────
TASK_ID="${SLURM_ARRAY_TASK_ID:?ERROR: SLURM_ARRAY_TASK_ID not set. Use --array or export it.}"

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "$N_TASKS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [1..${N_TASKS}]."
    echo "Available tasks (${N_VARIANTS} variants × ${N_STRETCHES} stretches):"
    t=1
    for si in "${!STRETCH_ARR[@]}"; do
        for vi in "${!VARIANT_NAMES[@]}"; do
            echo "  ${t}: ${VARIANT_NAMES[$vi]}  stretch=${STRETCH_ARR[$si]}  ${APPLY_ARGS[$vi]}"
            t=$((t + 1))
        done
    done
    exit 1
fi

IDX=$((TASK_ID - 1))
STRETCH_IDX=$((IDX / N_VARIANTS))
VARIANT_IDX=$((IDX % N_VARIANTS))
STRETCH="${STRETCH_ARR[$STRETCH_IDX]}"
VARIANT="${VARIANT_NAMES[$VARIANT_IDX]}"
VARIANT_ARGS="${APPLY_ARGS[$VARIANT_IDX]}"

if [ -n "$RUN_TAG" ]; then
    RUN_NAME="EXP_${RUN_TAG}_${VARIANT}_s${STRETCH}"
elif [ "$N_STRETCHES" -gt 1 ]; then
    RUN_NAME="EXP_${VARIANT}_s${STRETCH}"
else
    RUN_NAME="EXP_${VARIANT}"
fi

WORK_DIR="$(pwd)"
RUN_DIR="sweep/${RUN_NAME}"
ADV_DIR="${RUN_DIR}/adv_videos"
ADV_DIR_ABS="${WORK_DIR}/${ADV_DIR}"

echo "=============================================="
echo "Temporal Order Ablation"
echo "=============================================="
echo "  Variant       : ${VARIANT}"
echo "  Apply args    : ${VARIANT_ARGS}"
echo "  Array task    : ${TASK_ID} / ${N_TASKS}"
echo "  UAP           : ${UAP_PATH}"
echo "  Run tag       : ${RUN_TAG:-<none>}"
echo "  Stretch       : ${STRETCH}"
echo "  CRF           : ${CRF}"
echo "  Output        : ${RUN_DIR}"
echo "  Node          : $(hostname)"
echo "  GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start time    : $(date)"
echo "=============================================="
echo ""

if [ ! -f "$UAP_PATH" ]; then
    echo "ERROR: ${UAP_PATH} not found"
    exit 1
fi

mkdir -p "$RUN_DIR" "$ADV_DIR"

cat > "${RUN_DIR}/grid_config.txt" <<CFGEOF
experiment=temporal_ablation
run_name=${RUN_NAME}
uap_path=${UAP_PATH}
run_tag=${RUN_TAG}
stretch=${STRETCH}
crf=${CRF}
apply_args=${VARIANT_ARGS}
CFGEOF

# ── Apply UAP with ablation flags ─────────────────────────────────────────
echo ">>> Applying UAP (${VARIANT_ARGS}) ..."

srun python "${REPO_ROOT}/attack/apply_uap.py" \
    --uap "$UAP_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$ADV_DIR" \
    --stretch "$STRETCH" \
    --crf "$CRF" \
    ${VARIANT_ARGS}

echo "    Videos saved to ${ADV_DIR}"
echo ""

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

# ── Evaluate on Qwen3-VL ─────────────────────────────────────────────────
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

# ── Evaluate on LLaVA-OneVision ──────────────────────────────────────────
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

# ── Evaluate on VideoLLaMA3 ──────────────────────────────────────────────
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
echo "${RUN_NAME} finished at $(date)"
echo "Results in: ${RUN_DIR}/"
echo ""
echo "Interpretation (compare to in-order baseline):"
echo "  - Shuffle << baseline  => VLMs detect temporal ORDER"
echo "  - Static  << baseline  => Frame DIVERSITY needed"
echo "  - Reverse ~  baseline  => Directionality does not matter"
echo "=============================================="
