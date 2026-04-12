#!/bin/bash

#SBATCH -J grid_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --array=1-12
#SBATCH --output=sweep/logs/grid_%A_%a.out
#SBATCH --error=sweep/logs/grid_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════
# Focused Grid Search -- TRAIN ONLY (Phase 1)
# ══════════════════════════════════════════════════════════════════════════
# Trains 12 UAP configs around the S8 sweet spot without applying or
# evaluating on VLMs.  Use run_sweep_diagnose.sh afterwards to rank
# configs by surrogate diagnostics, then run_sweep_eval_top.sh on the
# best candidates.
#
# Submit all:  sbatch run_sweep_grid.sh
# Run one:     SLURM_ARRAY_TASK_ID=3 bash run_sweep_grid.sh
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

source ~/.bashrc
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
conda activate "${CONDA_ENV}"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p sweep/logs

# ── Paths ─────────────────────────────────────────────────────────────────
IMAGE_DIR="${IMAGE_DIR_NORMAL}"
TEMPORAL_PT="${REPO_ROOT}/accident_temporal.pt"

# ── Shared fixed parameters (from sweep findings) ────────────────────────
CLIP_MODELS=(ViT-L-14 EVA02-L-14 ViT-SO400M-14-SigLIP)
CLIP_PRETRAINED=(openai merged2b_s4b_b131k webli)
EPSILON=0.0627
N=32
MU=0.9
IMAGE_SIZE=448
DI_PROB=0.5
DI_SCALE_LOW=0.8
LAMBDA_NEG=0.5

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

# ── Per-run configurations ────────────────────────────────────────────────
# Format: RUN_NAME  ALPHA  LAMBDA_TRAJ  LAMBDA_TRANS  EPOCHS
#
# All configs use temporal losses (>=0.1 traj).  No-temporal ablation
# already covered by S5 in the original sweep.
#
# Group A: alpha=0.004, varied temporal (fill gaps around S8)
# Group B: finer alpha near S8 (0.0035, 0.0045)
# Group C: alpha=0.003, varied temporal
# Group D: alpha=0.005, unexplored midpoint

declare -A CONFIGS
CONFIGS[1]="G1_a4-low-temp         0.004  0.1   0.05   6"
CONFIGS[2]="G2_a4-low-hightrans    0.004  0.1   0.1    6"
CONFIGS[3]="G3_a4-near-s8-low      0.004  0.12  0.08   6"
CONFIGS[4]="G4_a4-near-s8-high     0.004  0.18  0.12   6"
CONFIGS[5]="G5_a4-mid-temp         0.004  0.2   0.1    6"
CONFIGS[6]="G6_a4-mid-hightrans    0.004  0.2   0.15   6"
CONFIGS[7]="G7_a35-mod-temporal    0.0035 0.15  0.1    6"
CONFIGS[8]="G8_a45-mod-temporal    0.0045 0.15  0.1    6"
CONFIGS[9]="G9_a3-mod-temporal     0.003  0.15  0.1    6"
CONFIGS[10]="G10_a3-low-temp       0.003  0.1   0.05   6"
CONFIGS[11]="G11_a5-mod-temporal   0.005  0.15  0.1    6"
CONFIGS[12]="G12_a5-low-temp       0.005  0.1   0.05   6"

# ── Select configuration based on array index ─────────────────────────────
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

read -r RUN_NAME ALPHA LAMBDA_TRAJ LAMBDA_TRANS EPOCHS \
    <<< "${CONFIGS[$TASK_ID]}"

RUN_DIR="sweep/${RUN_NAME}"
UAP_PATH="${RUN_DIR}/tti_uap.pt"

mkdir -p "$RUN_DIR"

echo "=============================================="
echo "Grid Run      : ${RUN_NAME}"
echo "Array Task    : ${TASK_ID}"
echo "Node          : $(hostname)"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time    : $(date)"
echo "----------------------------------------------"
echo "  alpha         : ${ALPHA}"
echo "  lambda_traj   : ${LAMBDA_TRAJ}"
echo "  lambda_trans  : ${LAMBDA_TRANS}"
echo "  epochs        : ${EPOCHS}"
echo "  (fixed) image_size=${IMAGE_SIZE} di_prob=${DI_PROB} di_scale_low=${DI_SCALE_LOW}"
echo "=============================================="
echo ""

# ══════════════════════════════════════════════════════════════════════════
# Train UAP (only step -- no apply, no VLM eval)
# ══════════════════════════════════════════════════════════════════════════
echo ">>> Training UAP for ${RUN_NAME} ..."

srun python "${REPO_ROOT}/attack/tti_attack.py" \
    --image_dir "$IMAGE_DIR" \
    --output "$UAP_PATH" \
    --clip_models "${CLIP_MODELS[@]}" \
    --clip_pretrained_list "${CLIP_PRETRAINED[@]}" \
    --target_texts "${TARGET_TEXTS[@]}" \
    --negative_texts "${NEGATIVE_TEXTS[@]}" \
    --accident_temporal "${TEMPORAL_PT}" \
    --lambda_neg "$LAMBDA_NEG" \
    --lambda_traj "$LAMBDA_TRAJ" \
    --lambda_trans "$LAMBDA_TRANS" \
    --epsilon "$EPSILON" \
    --alpha "$ALPHA" \
    --N "$N" \
    --image_size "$IMAGE_SIZE" \
    --device cuda \
    --epochs "$EPOCHS" \
    --mu "$MU" \
    --di_prob "$DI_PROB" \
    --di_scale_low "$DI_SCALE_LOW"

echo ""
echo "UAP saved to ${UAP_PATH}"

# Save config metadata for summarize.py
cat > "${RUN_DIR}/grid_config.txt" <<CFGEOF
alpha=${ALPHA}
lambda_traj=${LAMBDA_TRAJ}
lambda_trans=${LAMBDA_TRANS}
epochs=${EPOCHS}
image_size=${IMAGE_SIZE}
di_prob=${DI_PROB}
di_scale_low=${DI_SCALE_LOW}
lambda_neg=${LAMBDA_NEG}
CFGEOF

echo ""
echo "=============================================="
echo "Grid run ${RUN_NAME} finished at $(date)"
echo "=============================================="
