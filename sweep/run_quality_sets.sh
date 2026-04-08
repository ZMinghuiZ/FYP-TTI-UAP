#!/bin/bash

#SBATCH -J quality_sets
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sweep/logs/quality_sets_%j.out
#SBATCH --error=sweep/logs/quality_sets_%j.err

# ══════════════════════════════════════════════════════════════════════════
# Quality Metrics -- PSNR + SSIM with Customisable Directory Sets
# ══════════════════════════════════════════════════════════════════════════
#
# CPU-only job: PSNR uses OpenCV (C++), SSIM uses skimage (NumPy).
# No GPU needed -- saves queue time and allocation budget.
#
# Processes directories in parallel (one compute_quality.py per directory).
# Directories with an existing quality_metrics.csv are skipped unless
# FORCE_RECOMPUTE=1 is set.
#
# Usage:
#   bash run_quality_sets.sh SET_NAME [RUN_NAME]
#   sbatch run_quality_sets.sh SET_NAME [RUN_NAME]
#
# SET_NAME is one of:  tradeoff | stretch | all | full
# RUN_NAME defaults to G4_a4-near-s8-high
#
# Environment variables:
#   MAX_PARALLEL      -- max dirs processed concurrently  (default: 4)
#   WORKERS_PER_DIR   -- multiprocessing workers per dir  (default: 4)
#   SAMPLE_EVERY      -- compute on every Nth frame       (default: 10)
#   FORCE_RECOMPUTE   -- set to 1 to redo completed dirs  (default: 0)
#
# Examples:
#   bash run_quality_sets.sh tradeoff
#   bash run_quality_sets.sh all
#   SAMPLE_EVERY=1 bash run_quality_sets.sh full        # every frame
#   FORCE_RECOMPUTE=1 bash run_quality_sets.sh stretch  # redo all
# ══════════════════════════════════════════════════════════════════════════

set -euo pipefail

source ~/.bashrc
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"
conda activate "${CONDA_ENV}"
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p sweep/logs

# ── Paths & parallelism ──────────────────────────────────────────────────
CLEAN_DIR="${VIDEO_DIR_CLEAN}"
SUMMARY_CSV="sweep/quality_summary.csv"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
WORKERS_PER_DIR="${WORKERS_PER_DIR:-4}"
SAMPLE_EVERY="${SAMPLE_EVERY:-10}"
FORCE_RECOMPUTE="${FORCE_RECOMPUTE:-0}"

# ── Arguments ─────────────────────────────────────────────────────────────
SET_NAME="${1:-}"
RUN_NAME="${2:-G4_a4-near-s8-high}"
RUN_DIR="sweep/${RUN_NAME}"

if [ -z "$SET_NAME" ]; then
    echo "ERROR: No set name provided."
    echo ""
    echo "Usage: bash $0 SET_NAME [RUN_NAME]"
    echo ""
    echo "Available sets:"
    echo "  tradeoff  -- ASR-vs-quality tradeoff (F=24,32,64 x multiple scales)"
    echo "  stretch   -- stretch isolation (F=14..256 x scale=1.0)"
    echo "  all       -- tradeoff + stretch combined"
    echo "  full      -- every adv_videos_stretch* dir (glob)"
    echo ""
    echo "RUN_NAME defaults to G4_a4-near-s8-high"
    exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Run directory not found: ${RUN_DIR}"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# Define directory sets -- edit these arrays to customise
# ══════════════════════════════════════════════════════════════════════════

# Set A: ASR-vs-imperceptibility tradeoff curve
#   F=24,32,64 with all available scale values
TRADEOFF_DIRS=(
    "adv_videos_stretch32_scale0.5"
    "adv_videos_stretch32_scale0.625"
    "adv_videos_stretch32_scale0.75"
    "adv_videos_stretch32_scale1"
    "adv_videos_stretch64_scale0.5"
    "adv_videos_stretch64_scale0.625"
    "adv_videos_stretch64_scale0.75"
    "adv_videos_stretch64_scale1"
)

# Set B: stretch isolation at fixed scale=1.0
STRETCH_DIRS=(
    "adv_videos_stretch12_scale1"
    "adv_videos_stretch14_scale1"
    "adv_videos_stretch16_scale1"
    "adv_videos_stretch18_scale1"
    "adv_videos_stretch96_scale1"
    "adv_videos_stretch128_scale1"
    "adv_videos_stretch160_scale1"
    "adv_videos_stretch192_scale1"
    "adv_videos_stretch256_scale1"
)

# ── Build the selected directory list ─────────────────────────────────────
ADV_DIRS=()

case "$SET_NAME" in
    tradeoff)
        for d in "${TRADEOFF_DIRS[@]}"; do
            ADV_DIRS+=("${RUN_DIR}/${d}")
        done
        ;;
    stretch)
        for d in "${STRETCH_DIRS[@]}"; do
            ADV_DIRS+=("${RUN_DIR}/${d}")
        done
        ;;
    all)
        for d in "${TRADEOFF_DIRS[@]}" "${STRETCH_DIRS[@]}"; do
            ADV_DIRS+=("${RUN_DIR}/${d}")
        done
        ;;
    full)
        for d in "${RUN_DIR}"/adv_videos_stretch*; do
            [ -d "$d" ] && ADV_DIRS+=("$d")
        done
        ;;
    *)
        echo "ERROR: Unknown set '${SET_NAME}'."
        echo "Choose from: tradeoff | stretch | all | full"
        exit 1
        ;;
esac

# ── Validate directories & apply resume logic ────────────────────────────
VALID_DIRS=()
MISSING=0
SKIPPED_DONE=0
for d in "${ADV_DIRS[@]}"; do
    if [ ! -d "$d" ]; then
        echo "WARNING: Directory not found, skipping: $d"
        MISSING=$((MISSING + 1))
        continue
    fi
    if [ "$FORCE_RECOMPUTE" != "1" ] && [ -f "$d/quality_metrics.csv" ] \
       && [ -s "$d/quality_metrics.csv" ]; then
        echo "SKIP (already done): $d"
        SKIPPED_DONE=$((SKIPPED_DONE + 1))
        continue
    fi
    VALID_DIRS+=("$d")
done

echo ""
echo "=============================================="
echo "Quality Metrics (PSNR + SSIM) -- parallel"
echo "Set           : ${SET_NAME}"
echo "Run           : ${RUN_NAME}"
echo "Clean videos  : ${CLEAN_DIR}"
echo "Max parallel  : ${MAX_PARALLEL}"
echo "Workers/dir   : ${WORKERS_PER_DIR}"
echo "Sample every  : ${SAMPLE_EVERY} frame(s)"
echo "Requested     : ${#ADV_DIRS[@]} dirs"
echo "Already done  : ${SKIPPED_DONE} dirs (skipped)"
echo "To process    : ${#VALID_DIRS[@]} dirs"
if [ "$MISSING" -gt 0 ]; then
echo "Missing       : ${MISSING} dirs (skipped)"
fi
echo "Summary CSV   : ${SUMMARY_CSV}"
echo "Node          : $(hostname)"
echo "Start time    : $(date)"
echo "=============================================="
echo ""

if [ "${#VALID_DIRS[@]}" -eq 0 ]; then
    echo "Nothing to process (all directories already have quality_metrics.csv)."
    echo "Set FORCE_RECOMPUTE=1 to redo them."
    exit 0
fi

echo "Directories to process:"
for d in "${VALID_DIRS[@]}"; do
    echo "  $d"
done
echo ""

# ── Temp dir for per-directory summaries (merged at the end) ──────────────
TMP_DIR=$(mktemp -d)
LOG_DIR="sweep/logs/quality_${SET_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

START_EPOCH=$(date +%s)

# ── Launch parallel jobs ──────────────────────────────────────────────────
PIDS=()
DIR_NAMES=()

for i in "${!VALID_DIRS[@]}"; do
    d="${VALID_DIRS[$i]}"
    tmp_csv="${TMP_DIR}/summary_${i}.csv"
    log_file="${LOG_DIR}/$(basename "$d").log"

    python "${REPO_ROOT}/analysis/compute_quality.py" \
        --clean_dir "$CLEAN_DIR" \
        --adv_dir "$d" \
        --workers "$WORKERS_PER_DIR" \
        --sample_every "$SAMPLE_EVERY" \
        --summary "$tmp_csv" \
        > "$log_file" 2>&1 &

    PIDS+=($!)
    DIR_NAMES+=("$d")

    # Throttle to MAX_PARALLEL concurrent jobs
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        sleep 2
    done
done

# ── Wait for all jobs & collect exit codes ────────────────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  DONE  : ${DIR_NAMES[$i]}"
    else
        echo "  FAILED: ${DIR_NAMES[$i]}"
        echo "    log : ${LOG_DIR}/$(basename "${DIR_NAMES[$i]}").log"
        FAILED=$((FAILED + 1))
    fi
done
echo ""

# ── Merge per-directory summaries into main CSV ──────────────────────────
NEED_HEADER=true
if [ -f "$SUMMARY_CSV" ] && [ -s "$SUMMARY_CSV" ]; then
    NEED_HEADER=false
fi

MERGED=0
for f in "${TMP_DIR}"/summary_*.csv; do
    [ -f "$f" ] || continue
    if $NEED_HEADER; then
        cat "$f" >> "$SUMMARY_CSV"
        NEED_HEADER=false
    else
        tail -n +2 "$f" >> "$SUMMARY_CSV"
    fi
    MERGED=$((MERGED + 1))
done

END_EPOCH=$(date +%s)
ELAPSED=$(( END_EPOCH - START_EPOCH ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

echo "=============================================="
echo "Quality metrics complete for ${SET_NAME} at $(date)"
echo "Dirs processed: ${MERGED}/${#VALID_DIRS[@]}"
if [ "$FAILED" -gt 0 ]; then
echo "Dirs failed   : ${FAILED}"
echo "Logs          : ${LOG_DIR}/"
fi
echo "Summary CSV   : ${SUMMARY_CSV}"
echo "Elapsed       : ${ELAPSED_MIN}m ${ELAPSED}s"
echo "=============================================="
