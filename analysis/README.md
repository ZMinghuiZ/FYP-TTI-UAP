# analysis/ — Post-hoc Analysis and Metrics

This directory contains scripts that consume CSV outputs from the `evaluation/` pipeline and compute aggregate metrics, statistical tests, and quality measurements.

## Scripts

### `analyse_responses.py`

Mine model answer text from evaluation CSVs for keyword patterns. Counts temporal-abnormal, accident (with negation awareness), event-descriptor, and benign keywords. Supports pairwise Mann–Whitney U tests between conditions.

**CLI:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--sweep_dir` | `sweep/` | Root directory to scan for `*_results_*.csv` |
| `--csv_files` | `None` | Explicit CSV paths (bypasses sweep scan) |
| `--configs` | `None` | Filter to specific configs (default: all found) |
| `--output_csv` | `None` | Comparative summary CSV |
| `--output_details` | `None` | Per-video detail CSV |
| `--verbose` | off | Print extra detail |

**Input CSV columns:** `model_answer`, `prediction`, `status`, `label`, `filename`.

**Output CSV columns (`--output_csv`):** `model`, `name`, `n`, `asr`, `avg_words`, `avg_len`, `avg_temporal`, `avg_accident_affirmed`, `avg_accident_negated`, `avg_event`, `avg_benign`, `per100w_temporal`, `per100w_accident`, `per100w_event`, `has_temporal_frac`, `has_accident_frac`, `has_event_frac`, `has_benign_frac`.

**Output CSV columns (`--output_details`):** Per-video rows with keyword match lists and truncated answer text.

**Statistics:** Mann–Whitney U between all condition pairs per model (requires *n* ≥ 20 in both groups).

---

### `analyse_event_verify.py`

Compute per-event false-positive rates from event verification CSVs, with Fisher's exact pairwise tests and Holm–Bonferroni correction. Includes temporal vs spatial FPR-delta narrative analysis.

**CLI:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--sweep_dir` | `sweep/EXP_event_verify/` | Sweep directory for auto-discovery |
| `--csv_files` | `None` | Explicit CSV paths |
| `--output_csv` | `None` | Output summary CSV |
| `--alpha` | `0.05` | Significance level |

**Input CSV columns:** `collision`, `loss_of_control`, `fire_smoke`, `overturn`, `pedestrian`, `visual_artifacts`, `label`. Values are `yes`/`no`; rows with `error` in any event column are dropped.

**Output CSV columns:** `model`, `condition`, `n`, `fpr_collision`, `fpr_loss_of_control`, `fpr_fire_smoke`, `fpr_overturn`, `fpr_pedestrian`, `fpr_visual_artifacts`, `n_yes_collision`, …, `composite_fpr`, `composite_n_yes`.

**Composite FPR:** Fraction of videos with "yes" on any of the five hallucination events (excludes `visual_artifacts`).

**Statistics:** Fisher's exact test (two-sided) between conditions per event, with Holm–Bonferroni correction across all tests per model.

---

### `summarize_accident_eval.py`

Summarize binary evaluation on real accident videos (ground-truth positive). Computes true-positive rate (TPR) and false-negative rate (FNR) with configurable ambiguous-prediction handling.

**CLI:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--results_dir` | *required* | Directory to scan for result CSVs |
| `--glob` | `*_results_*.csv` | Glob pattern for CSV discovery |
| `--ambiguous_policy` | `miss` | `miss` (count as FN) or `exclude` (drop from denominator) |
| `--output_csv` | `None` | Summary CSV |

**Input CSV columns:** `prediction` (primary), with fallbacks to `status` and `model_answer` for error detection.

**Output CSV columns:** `scope`, `name`, `ambiguous_policy`, `tpr`, `fnr`, `tp`, `fn`, `denominator`, `ambiguous`, `errors`, `yes`, `no`, `total_rows`.

**Stdout:** Per-file, per-model, and overall micro-averaged tables.

---

### `compute_quality.py`

Compare adversarial videos to their clean counterparts frame-by-frame, computing PSNR and SSIM. Clean videos are matched by stripping the `_adv` suffix from adversarial filenames.

**CLI:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--clean_dir` | *required* | Root directory of clean videos (recursive) |
| `--adv_dir` | *required* (one or more) | Adversarial video directories (top-level files only) |
| `--workers` | `4` | Parallel workers per directory |
| `--sample_every` | `1` | Sample every Nth frame |
| `--summary` | `None` | Append per-directory aggregate to this CSV |

**Per-directory output:** `{adv_dir}/quality_metrics.csv` with columns `filename`, `psnr_mean`, `psnr_std`, `ssim_mean`, `ssim_std`, `num_frames`, `total_frames`.

**Summary CSV columns (`--summary`):** `directory`, `num_videos`, `psnr_mean`, `psnr_std`, `ssim_mean`, `ssim_std`.

**Dependencies:** Requires `scikit-image` for SSIM (in addition to OpenCV for PSNR).

## Typical Workflow

```bash
# 1. Analyse keyword patterns in descriptive/binary eval results
python analysis/analyse_responses.py \
    --sweep_dir sweep/ \
    --output_csv experiment_result/response_analysis.csv

# 2. Analyse event-verification false-positive rates
python analysis/analyse_event_verify.py \
    --sweep_dir sweep/EXP_event_verify/ \
    --output_csv experiment_result/event_verify_analysis.csv

# 3. Summarize accident-positive eval (ground-truth videos)
python analysis/summarize_accident_eval.py \
    --results_dir sweep/EXP_accident/ \
    --output_csv experiment_result/accident_summary.csv

# 4. Compute video quality metrics
python analysis/compute_quality.py \
    --clean_dir /path/to/clean/videos \
    --adv_dir sweep/G4_a4-near-s8-high/adv_videos \
    --summary sweep/quality_summary.csv
```
