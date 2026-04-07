#!/bin/bash

#SBATCH -J grid_diagnose
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=192G
#SBATCH --time=06:00:00
#SBATCH --output=sweep/logs/diagnose_%j.out
#SBATCH --error=sweep/logs/diagnose_%j.err

# ══════════════════════════════════════════════════════════════════════════
# Batch Diagnose All Sweep UAPs (Phase 2)
# ══════════════════════════════════════════════════════════════════════════
# Runs diagnose_uap.py on every sweep/*/btc_uap.pt and saves output to
# sweep/*/diagnose_output.txt.  Then calls sweep/summarize.py --diagnose
# to rank configs by composite transferability score.
#
# Submit:   sbatch run_sweep_diagnose.sh
# Or local: bash run_sweep_diagnose.sh
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

source ~/.bashrc
conda activate videollama

SCRIPT_DIR="/home/z/zminghui/ult_attack"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p sweep/logs

# ── Paths ─────────────────────────────────────────────────────────────────
IMAGE_DIR="/home/z/zminghui/traffic_images/normal_train"
TEMPORAL_PT="${SCRIPT_DIR}/accident_temporal.pt"

# ── CLIP models (same as training) ────────────────────────────────────────
CLIP_MODELS=(ViT-L-14 EVA02-L-14 ViT-SO400M-14-SigLIP)
CLIP_PRETRAINED=(openai merged2b_s4b_b131k webli)

TARGET_TEXTS=(
    "Yes, there is a road accident in the video"
    "A traffic accident or anomaly is visible"
    "There are vehicles involved in an accident"
    "A road accident with vehicle collision"
)
NEGATIVE_TEXTS=(
    "normal traffic flow on the road"
    "safe driving on a clear highway"
    "cars driving normally in traffic"
    "a peaceful road with vehicles"
)

echo "=============================================="
echo "Batch UAP Diagnosis"
echo "Node          : $(hostname)"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time    : $(date)"
echo "=============================================="
echo ""

TOTAL=0
DIAGNOSED=0
SKIPPED=0

for RUN_DIR in sweep/S*/ sweep/G*/; do
    [ -d "$RUN_DIR" ] || continue
    UAP_PATH="${RUN_DIR}btc_uap.pt"

    if [ ! -f "$UAP_PATH" ]; then
        echo "SKIP ${RUN_DIR} -- no btc_uap.pt found"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    TOTAL=$((TOTAL + 1))
    RUN_NAME=$(basename "$RUN_DIR")
    OUTPUT="${RUN_DIR}diagnose_output.txt"

    echo "----------------------------------------------"
    echo ">>> Diagnosing ${RUN_NAME} ..."
    echo "    UAP: ${UAP_PATH}"

    python "${SCRIPT_DIR}/diagnose_uap.py" \
        --uap "$UAP_PATH" \
        --image_dir "$IMAGE_DIR" \
        --max_images 100 \
        --clip_models "${CLIP_MODELS[@]}" \
        --clip_pretrained_list "${CLIP_PRETRAINED[@]}" \
        --target_texts "${TARGET_TEXTS[@]}" \
        --negative_texts "${NEGATIVE_TEXTS[@]}" \
        --accident_temporal "$TEMPORAL_PT" \
        --device cuda \
        2>&1 | tee "$OUTPUT"

    DIAGNOSED=$((DIAGNOSED + 1))
    echo ""
    echo "    Output saved to ${OUTPUT}"
    echo ""
done

echo "=============================================="
echo "Diagnosis complete: ${DIAGNOSED}/${TOTAL} runs diagnosed, ${SKIPPED} skipped"
echo ""

# ── Summarize with diagnostic ranking ─────────────────────────────────────
echo ">>> Generating ranked summary ..."
python sweep/summarize.py --diagnose --csv

echo ""
echo "=============================================="
echo "All done at $(date)"
echo "=============================================="
