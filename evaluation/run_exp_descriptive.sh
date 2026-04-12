#!/bin/bash

#SBATCH -J descriptive_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=sweep/logs/descriptive_%A_%a.out
#SBATCH --error=sweep/logs/descriptive_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════
# Experiment: Descriptive VLM Evaluation  (parallel via SLURM array)
# ══════════════════════════════════════════════════════════════════════════
# Asks VLMs to describe what happens in videos step by step (instead of
# yes/no), to reveal whether they perceive temporal events in adversarial
# videos that they don't see in clean ones.
#
# Each array task = one (model, condition) pair on its own GPU:
#
#   Task  1 : InternVL3      × G4_temporal       (~1-2h)
#   Task  2 : InternVL3      × S13_no_temporal   (~1-2h)
#   Task  3 : InternVL3      × clean             (~1-2h)
#   Task  4 : Qwen3-VL       × G4_temporal       (~1-1.5h)
#   Task  5 : Qwen3-VL       × S13_no_temporal   (~1-1.5h)
#   Task  6 : Qwen3-VL       × clean             (~1-1.5h)
#   Task  7 : LLaVA-OV       × G4_temporal       (~0.5-1h)
#   Task  8 : LLaVA-OV       × S13_no_temporal   (~0.5-1h)
#   Task  9 : LLaVA-OV       × clean             (~0.5-1h)
#   Task 10 : VideoLLaMA3    × G4_temporal       (~0.5-1h)
#   Task 11 : VideoLLaMA3    × S13_no_temporal   (~0.5-1h)
#   Task 12 : VideoLLaMA3    × clean             (~0.5-1h)
#
# Usage:
#   sbatch --array=1-12 run_exp_descriptive.sh       # all 12 in parallel
#   sbatch --array=1-6  run_exp_descriptive.sh       # InternVL3 + Qwen3 only
#   sbatch --array=1-3  run_exp_descriptive.sh       # InternVL3 only
#   sbatch --array=4-6  run_exp_descriptive.sh       # Qwen3-VL only
#   sbatch --array=7-9  run_exp_descriptive.sh       # LLaVA-OV only
#   sbatch --array=10-12 run_exp_descriptive.sh      # VideoLLaMA3 only
#   sbatch --array=1,4,7,10 run_exp_descriptive.sh   # G4 only, all models
#   SLURM_ARRAY_TASK_ID=1 bash run_exp_descriptive.sh  # local test
#
# After ALL jobs finish, run the analysis:
#   python analysis/analyse_responses.py --sweep_dir sweep/EXP_descriptive/ \
#       --output_csv sweep/EXP_descriptive/descriptive_comparison.csv \
#       --output_details sweep/EXP_descriptive/descriptive_per_video.csv \
#       --verbose
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

source ~/.bashrc
source "${SLURM_SUBMIT_DIR}/config.sh"
conda activate "${CONDA_ENV}"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p sweep/logs

# ── Video directories ──────────────────────────────────────────────────
WORK_DIR="$(pwd)"
G4_ADV_DIR="${WORK_DIR}/sweep/G4_a4-near-s8-high/adv_videos"
S13_ADV_DIR="${WORK_DIR}/sweep/S13_no-temporal-small/adv_videos"
CLEAN_DIR="${VIDEO_DIR_CLEAN}"

RESULTS_DIR="${WORK_DIR}/sweep/EXP_descriptive"
mkdir -p "${RESULTS_DIR}"

# ── Task matrix: (model, condition, video_dir) ─────────────────────────
MODELS=(     "internvl"      "internvl"        "internvl"    \
             "qwen3"         "qwen3"           "qwen3"       \
             "llava"         "llava"           "llava"       \
             "videollama3"   "videollama3"     "videollama3" )
CONDITIONS=( "G4_temporal"   "S13_no_temporal" "clean"       \
             "G4_temporal"   "S13_no_temporal" "clean"       \
             "G4_temporal"   "S13_no_temporal" "clean"       \
             "G4_temporal"   "S13_no_temporal" "clean"       )
VIDEO_DIRS=( "${G4_ADV_DIR}" "${S13_ADV_DIR}"  "${CLEAN_DIR}" \
             "${G4_ADV_DIR}" "${S13_ADV_DIR}"  "${CLEAN_DIR}" \
             "${G4_ADV_DIR}" "${S13_ADV_DIR}"  "${CLEAN_DIR}" \
             "${G4_ADV_DIR}" "${S13_ADV_DIR}"  "${CLEAN_DIR}" )

TASK_ID="${SLURM_ARRAY_TASK_ID:?ERROR: SLURM_ARRAY_TASK_ID not set. Use --array or export it.}"
N_TASKS="${#MODELS[@]}"

if [ "$TASK_ID" -lt 1 ] || [ "$TASK_ID" -gt "$N_TASKS" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [1..${N_TASKS}]."
    echo "Available tasks:"
    for i in $(seq 0 $((N_TASKS - 1))); do
        echo "  $((i + 1)): ${MODELS[$i]} × ${CONDITIONS[$i]}"
    done
    exit 1
fi

IDX=$((TASK_ID - 1))
MODEL="${MODELS[$IDX]}"
CONDITION="${CONDITIONS[$IDX]}"
VIDEO_DIR="${VIDEO_DIRS[$IDX]}"
OUT_DIR="${RESULTS_DIR}/${CONDITION}_${MODEL}"
mkdir -p "${OUT_DIR}"

echo "=============================================="
echo "Experiment: Descriptive VLM Evaluation"
echo "=============================================="
echo "  Task        : ${TASK_ID}/${N_TASKS}"
echo "  Model       : ${MODEL}"
echo "  Condition   : ${CONDITION}"
echo "  Video dir   : ${VIDEO_DIR}"
echo "  Output dir  : ${OUT_DIR}"
echo "  Node        : $(hostname)"
echo "  GPU         : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Start time  : $(date)"
echo "=============================================="
echo ""

# ══════════════════════════════════════════════════════════════════════════
# Run single (model, condition) eval
# ══════════════════════════════════════════════════════════════════════════

srun python "${REPO_ROOT}/evaluation/eval_descriptive.py" \
    --model "$MODEL" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUT_DIR" \
    --label "$CONDITION"

echo ""
echo "=============================================="
echo "Task ${TASK_ID} (${MODEL} × ${CONDITION}) finished at $(date)"
echo "Results in: ${OUT_DIR}/"
echo ""
echo "When ALL array tasks are done, run analysis:"
echo "  python analysis/analyse_responses.py --sweep_dir sweep/EXP_descriptive/ \\"
echo "    --output_csv sweep/EXP_descriptive/descriptive_comparison.csv \\"
echo "    --output_details sweep/EXP_descriptive/descriptive_per_video.csv \\"
echo "    --verbose"
echo "=============================================="
