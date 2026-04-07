#!/bin/bash

#SBATCH -J uap_attack
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=logs/btc_%j.out
#SBATCH --error=logs/btc_%j.err


source ~/.bashrc
conda activate videollama

cd $SLURM_SUBMIT_DIR
mkdir -p logs

echo "=============================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Job Name     : $SLURM_JOB_NAME"
echo "Node         : $(hostname)"
echo "GPU          : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start time   : $(date)"
echo "=============================================="
echo ""

# ══════════════════════════════════════════════════════════════════════════
# Step 0: Pre-compute accident temporal templates
# ══════════════════════════════════════════════════════════════════════════
# Extracts per-model trajectory templates (N, D) and transition similarity
# patterns (N-1,) from real accident videos.  Impact-weighted sampling
# concentrates frames near the collision point.
if [ ! -f ./accident_temporal.pt ]; then
    echo ">>> Step 0: Pre-computing accident temporal templates ..."
    srun python precompute_accident_temporal.py \
        --video_dir /home/z/zminghui/videos/acc_train \
        --output ./accident_temporal.pt \
        --N 32 \
        --clip_models \
            ViT-L-14 \
            EVA02-L-14 \
            ViT-SO400M-14-SigLIP \
        --clip_pretrained_list \
            openai \
            merged2b_s4b_b131k \
            webli \
        --impact_weight 0.4 \
        --device cuda
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Train TTI-UAP (Temporal Trajectory Injection)
# ══════════════════════════════════════════════════════════════════════════
# Surrogate ensemble:
#   ViT-L-14 (openai)           -- strong general-purpose CLIP
#   EVA02-L-14 (merged2b)       -- close to InternViT used by InternVL3
#   ViT-SO400M-14-SigLIP (webli) -- sigmoid loss, close to Qwen3-VL
#
# Three losses:
#   L_clip  : per-frame targeted semantic shift toward accident text
#   L_traj  : trajectory matching against accident temporal template
#   L_trans : inter-frame transition pattern matching
echo ">>> Step 1: Training TTI-UAP ..."
srun python btc_attack.py \
    --image_dir /home/z/zminghui/traffic_images/normal_train \
    --output ./btc_uap.pt \
    --clip_models \
        ViT-L-14 \
        EVA02-L-14 \
        ViT-SO400M-14-SigLIP \
    --clip_pretrained_list \
        openai \
        merged2b_s4b_b131k \
        webli \
    --target_texts \
        "Yes, there is a road accident in the video" \
        "A traffic accident or anomaly is visible" \
        "There are vehicles involved in an accident" \
        "A road accident with vehicle collision" \
    --negative_texts \
        "normal traffic flow on the road" \
        "safe driving on a clear highway" \
        "cars driving normally in traffic" \
        "a peaceful road with vehicles" \
    --accident_temporal ./accident_temporal.pt \
    --lambda_neg 0.5 \
    --lambda_traj 0.3 \
    --lambda_trans 0.2 \
    --epsilon 0.0627 \
    --alpha 0.0078 \
    --N 32 \
    --image_size 448 \
    --device cuda \
    --epochs 10 \
    --mu 0.9 \
    --di_prob 0.5

# ══════════════════════════════════════════════════════════════════════════
# Step 1b: Diagnose UAP effectiveness (semantic + temporal)
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 1b: Diagnosing UAP effectiveness ..."
srun python diagnose_uap.py \
    --uap ./btc_uap.pt \
    --image_dir /home/z/zminghui/traffic_images/normal_train \
    --clip_models \
        ViT-L-14 \
        EVA02-L-14 \
        ViT-SO400M-14-SigLIP \
    --clip_pretrained_list \
        openai \
        merged2b_s4b_b131k \
        webli \
    --target_texts \
        "Yes, there is a road accident in the video" \
        "A traffic accident or anomaly is visible" \
    --negative_texts \
        "normal traffic flow on the road" \
        "safe driving on a clear highway" \
    --accident_temporal ./accident_temporal.pt \
    --max_images 200 \
    --device cuda

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Apply UAP to videos (LOSSLESS encoding)
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo ">>> Step 2: Applying UAP to videos (lossless) ..."
srun python apply_uap.py \
    --uap ./btc_uap.pt \
    --video_dir /home/z/zminghui/videos/eval/clean \
    --output_dir /home/z/zminghui/videos/target_adv_clean_v12 \
    --stretch 4 \
    --lossless

echo ""
echo "=============================================="
echo "Finished at  : $(date)"
echo "=============================================="
