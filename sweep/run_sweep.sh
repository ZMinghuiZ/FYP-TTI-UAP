#!/bin/bash

#SBATCH -J sweep_transfer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --array=1-10
#SBATCH --output=sweep/logs/sweep_%A_%a.out
#SBATCH --error=sweep/logs/sweep_%A_%a.err

# ══════════════════════════════════════════════════════════════════════════
# Hyperparameter Sweep for Transferability Exploration
# ══════════════════════════════════════════════════════════════════════════
# Submit with:  sbatch run_sweep.sh
# Or run one:   SLURM_ARRAY_TASK_ID=1 bash run_sweep.sh
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

source ~/.bashrc
source "${SLURM_SUBMIT_DIR}/config.sh"
conda activate "${CONDA_ENV}"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p sweep/logs

# ── Paths (edit these to match your cluster) ──────────────────────────────
IMAGE_DIR="${IMAGE_DIR_NORMAL}"
VIDEO_DIR="${VIDEO_DIR_CLEAN}"
TEMPORAL_PT="${REPO_ROOT}/accident_temporal.pt"

QWEN_MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
LLAVA_MODEL="llava-hf/llava-onevision-qwen2-7b-ov-hf"
VIDEOLLAMA3_MODEL="DAMO-NLP-SG/VideoLLaMA3-7B"
INTERNVL_MODEL="OpenGVLab/InternVL3-38B"

# ── Shared baseline parameters ────────────────────────────────────────────
CLIP_MODELS=(ViT-L-14 EVA02-L-14 ViT-SO400M-14-SigLIP)
CLIP_PRETRAINED=(openai merged2b_s4b_b131k webli)
EPSILON=0.0627
N=32
MU=0.9
STRETCH=4

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
# Each run overrides only the parameters that differ from baseline.
# Format: RUN_NAME  ALPHA  IMAGE_SIZE  DI_PROB  DI_SCALE_LOW  LAMBDA_NEG  LAMBDA_TRAJ  LAMBDA_TRANS  EPOCHS

declare -A CONFIGS
CONFIGS[1]="S1_wider-di           0.0078 448 0.5 0.5  0.5 0.3  0.2  6"
CONFIGS[2]="S2_wider-di-highprob  0.0078 448 0.7 0.5  0.5 0.3  0.2  6"
CONFIGS[3]="S3_mid-resolution     0.0078 384 0.5 0.8  0.5 0.3  0.2  6"
CONFIGS[4]="S4_mid-res-wider-di   0.0078 384 0.5 0.5  0.5 0.3  0.2  6"
CONFIGS[5]="S5_no-temporal        0.0078 448 0.5 0.8  0.5 0.0  0.0  6"
CONFIGS[6]="S6_low-temporal       0.0078 448 0.5 0.8  0.5 0.1  0.05 6"
CONFIGS[7]="S7_small-step         0.004  448 0.5 0.8  0.5 0.3  0.2  6"
CONFIGS[8]="S8_small-mod-temporal 0.004  448 0.5 0.8  0.5 0.15 0.1  6"
CONFIGS[9]="S9_smaller-step       0.003  448 0.5 0.8  0.5 0.3  0.2  6"
CONFIGS[10]="S10_small-wider-di   0.004  448 0.5 0.7  0.5 0.3  0.2  6"

# ── Select configuration based on array index ─────────────────────────────
TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"

read -r RUN_NAME ALPHA IMAGE_SIZE DI_PROB DI_SCALE_LOW LAMBDA_NEG LAMBDA_TRAJ LAMBDA_TRANS EPOCHS \
    <<< "${CONFIGS[$TASK_ID]}"

WORK_DIR="$(pwd)"
RUN_DIR="sweep/${RUN_NAME}"
ADV_DIR="${RUN_DIR}/adv_videos"
UAP_PATH="${RUN_DIR}/tti_uap.pt"
EVAL_CWD="${WORK_DIR}/${RUN_DIR}"
ADV_DIR_ABS="${WORK_DIR}/${ADV_DIR}"

mkdir -p "$RUN_DIR" "$ADV_DIR"

echo "=============================================="
echo "Sweep Run     : ${RUN_NAME}"
echo "Array Task    : ${TASK_ID}"
echo "Node          : $(hostname)"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time    : $(date)"
echo "----------------------------------------------"
echo "  alpha         : ${ALPHA}"
echo "  image_size    : ${IMAGE_SIZE}"
echo "  di_prob       : ${DI_PROB}"
echo "  di_scale_low  : ${DI_SCALE_LOW}"
echo "  lambda_neg    : ${LAMBDA_NEG}"
echo "  lambda_traj   : ${LAMBDA_TRAJ}"
echo "  lambda_trans  : ${LAMBDA_TRANS}"
echo "  epochs        : ${EPOCHS}"
echo "=============================================="
echo ""

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Train UAP
# ══════════════════════════════════════════════════════════════════════════
echo ">>> Step 1: Training UAP for ${RUN_NAME} ..."

TEMPORAL_ARGS=""
if [ "$LAMBDA_TRAJ" != "0.0" ] || [ "$LAMBDA_TRANS" != "0.0" ]; then
    TEMPORAL_ARGS="--accident_temporal ${TEMPORAL_PT}"
fi

srun python "${REPO_ROOT}/attack/tti_attack.py" \
    --image_dir "$IMAGE_DIR" \
    --output "$UAP_PATH" \
    --clip_models "${CLIP_MODELS[@]}" \
    --clip_pretrained_list "${CLIP_PRETRAINED[@]}" \
    --target_texts "${TARGET_TEXTS[@]}" \
    --negative_texts "${NEGATIVE_TEXTS[@]}" \
    $TEMPORAL_ARGS \
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

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Apply UAP to videos
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: Applying UAP to videos ..."

srun python "${REPO_ROOT}/attack/apply_uap.py" \
    --uap "$UAP_PATH" \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$ADV_DIR" \
    --stretch "$STRETCH"

echo ""
echo "Adversarial videos saved to ${ADV_DIR}"

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Evaluate on InternVL3
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 3: Evaluating on InternVL3 ..."

cd "$EVAL_CWD"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'internVL'))
os.chdir('${EVAL_CWD}')
from eval_ori import main
main('${INTERNVL_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"

echo ""
echo "InternVL3 evaluation complete. Results in ${RUN_DIR}/"

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Evaluate on Qwen3-VL
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 4: Evaluating on Qwen3-VL ..."

cd "$EVAL_CWD"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'qwen3'))
os.chdir('${EVAL_CWD}')
from eval_ori import main
main('${QWEN_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"

echo ""
echo "Qwen3-VL evaluation complete. Results in ${RUN_DIR}/"

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Evaluate on LLaVA-OneVision
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 5: Evaluating on LLaVA-OneVision ..."

cd "$EVAL_CWD"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'llava_onevision'))
os.chdir('${EVAL_CWD}')
from eval_ori import main
main('${LLAVA_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"

echo ""
echo "LLaVA-OneVision evaluation complete. Results in ${RUN_DIR}/"

# ══════════════════════════════════════════════════════════════════════════
# Step 6: Evaluate on VideoLLaMA3
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 6: Evaluating on VideoLLaMA3 ..."

cd "$EVAL_CWD"
srun python -c "
import sys, os
sys.path.insert(0, os.path.join('${REPO_ROOT}', 'evaluation', 'videollama3'))
os.chdir('${EVAL_CWD}')
from eval_ori import main
main('${VIDEOLLAMA3_MODEL}', '${ADV_DIR_ABS}')
"
cd "$WORK_DIR"

echo ""
echo "VideoLLaMA3 evaluation complete. Results in ${RUN_DIR}/"

# ══════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "=============================================="
echo "Sweep run ${RUN_NAME} finished at $(date)"
echo "=============================================="
