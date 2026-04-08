# config.sh -- edit these paths for your environment
#
# Auto-detects REPO_ROOT from the location of this file.
# All run_*.sh scripts source this file to get consistent paths.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT

export DATA_ROOT="${DATA_ROOT:-/home/z/zminghui}"
export VIDEO_DIR_CLEAN="${VIDEO_DIR_CLEAN:-${DATA_ROOT}/videos/eval/clean}"
export VIDEO_DIR_ACC_TRAIN="${VIDEO_DIR_ACC_TRAIN:-${DATA_ROOT}/videos/acc_train}"
export IMAGE_DIR_NORMAL="${IMAGE_DIR_NORMAL:-${DATA_ROOT}/traffic_images/normal_train}"
export ADV_OUTPUT_DIR="${ADV_OUTPUT_DIR:-${DATA_ROOT}/videos/target_adv_clean_v12}"
export CONDA_ENV="${CONDA_ENV:-videollama}"
