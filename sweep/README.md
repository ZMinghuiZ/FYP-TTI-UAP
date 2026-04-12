# sweep/ — Hyperparameter Sweep Utilities

This directory manages hyperparameter sweeps for UAP training, application, post-processing, and result aggregation. Sweep runs are stored as subdirectories (`S1_…`, `G1_…`, etc.) under `sweep/`, each containing a trained UAP, adversarial videos, and per-model evaluation outputs.

## Workflow

```
1. Train UAPs            run_sweep.sh / run_sweep_grid.sh
                                ↓
2. Apply & Evaluate      run_sweep_eval_top.sh
                                ↓
3. Stretch / Scale       run_sweep_stretch_scale.sh
   variants              run_stretch_parallel.sh
                         run_stretch_scale_parallel.sh
                                ↓
4. Post-processing       run_sweep_postprocess.sh
   robustness
                                ↓
5. Quality metrics       run_quality_sets.sh
                                ↓
6. Aggregate results     summarize.py / summarize_apply.py
                         summarize_quality.py
                         summarize_temporal_ablation.py
```

## Shell Scripts

### `run_sweep.sh`

Full pipeline per sweep config: train UAP → apply (stretch 4) → evaluate all four VLMs.

| SLURM Config | Value |
|-------------|-------|
| Array | `1-10` (configs `S1`–`S10`) |
| GPU | `h100-96:1` |
| Memory | `128G` |
| Time | `48:00:00` |

```bash
sbatch sweep/run_sweep.sh
```

### `run_sweep_grid.sh`

Train-only grid search: 12 configs (`G1`–`G12`) with varying hyperparameters. Writes `grid_config.txt` for downstream summarisers.

| SLURM Config | Value |
|-------------|-------|
| Array | `1-12` |
| GPU | `h100-96:1` |
| Time | `04:00:00` |

```bash
sbatch sweep/run_sweep_grid.sh
```

### `run_sweep_eval_top.sh`

Apply and evaluate selected top configs. Reads run names from `sweep/eval_top_list.txt` (one per line) or accepts a positional argument.

```bash
# Single run
sbatch sweep/run_sweep_eval_top.sh G4_a4-near-s8-high

# Array from list file
sbatch --array=1-N sweep/run_sweep_eval_top.sh
```

### `run_sweep_stretch_scale.sh`

Loop over stretch × scale combinations for a given run. Default grid: stretches `{14, 16, 18}` × scales `{1, 0.75, 0.5}` (9 combos per run).

```bash
sbatch sweep/run_sweep_stretch_scale.sh G4_a4-near-s8-high
```

### `run_stretch_parallel.sh`

Parallel stretch evaluation: one array task per stretch value from `STRETCHES=(4 12 24 32 64)`, fixed scale 1.

```bash
sbatch --array=1-5 sweep/run_stretch_parallel.sh G4_a4-near-s8-high
```

### `run_stretch_scale_parallel.sh`

Parallel stretch × scale evaluation: 6 tasks mapping to `(stretch, scale)` pairs at `{32, 64} × {0.5, 0.625, 0.75}`.

```bash
sbatch --array=1-6 sweep/run_stretch_scale_parallel.sh
```

### `run_sweep_postprocess.sh`

Apply 5 post-processing presets (mild, moderate, heavy, noise-only, filter-only) to existing adversarial videos, then evaluate each. Reads source directories from `sweep/postprocess_list.txt` or a positional argument.

```bash
sbatch --array=1-N sweep/run_sweep_postprocess.sh
```

### `run_quality_sets.sh`

CPU-oriented batch job for computing PSNR/SSIM via `analysis/compute_quality.py` over named sets of adversarial video directories.

```bash
# Predefined sets: tradeoff, stretch, all, full
sbatch sweep/run_quality_sets.sh tradeoff G4_a4-near-s8-high
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_PARALLEL` | `4` | Concurrent quality jobs |
| `WORKERS_PER_DIR` | `4` | Workers per directory |
| `SAMPLE_EVERY` | `10` | Frame sampling interval |
| `FORCE_RECOMPUTE` | `0` | Recompute existing metrics |

## Aggregation Scripts

### `summarize.py`

Scan `sweep/` for run directories (`S*`, `G*`), read ASR from `{model}_summary_metrics.txt` or `*_results_*.csv`, and print a comparative table. Optionally includes diagnostic scores.

```bash
python sweep/summarize.py --csv           # writes sweep/summary.csv
python sweep/summarize.py --csv --diagnose # includes diagnose_output.txt data
```

### `summarize_apply.py`

Aggregate ASR across all `adv_videos_stretch*_scale*` directories under sweep runs. Supports filtering by run name and result type (stretch-scale vs post-process).

```bash
python sweep/summarize_apply.py --csv                        # all runs
python sweep/summarize_apply.py --csv --run G4_a4-near-s8-high
python sweep/summarize_apply.py --csv --type postprocess     # post-process only
python sweep/summarize_apply.py --csv --sort mean            # sort by mean ASR
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | off | Write `sweep/summary_apply.csv` |
| `--run` | all | Filter to specific run name(s) |
| `--run_list` | `None` | File with run names (one per line) |
| `--type` | `all` | `all`, `stretch_scale`, or `postprocess` |
| `--sort` | `run` | Sort by `run`, `mean`, or `stretch` |

### `summarize_quality.py`

Aggregate per-video `quality_metrics.csv` from stretch × scale directories into a single table, optionally merged with ASR data.

```bash
python sweep/summarize_quality.py \
    --run_dir sweep/G4_a4-near-s8-high \
    --asr sweep/summary_G4_stretch_scale.csv \
    --output sweep/quality_asr_merged.csv
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--run_dir` | `sweep/G4_a4-near-s8-high` | Run directory with `adv_videos_*` subdirs |
| `--asr` | `sweep/summary_G4_stretch_scale.csv` | ASR CSV to merge (optional) |
| `--output` | `sweep/quality_asr_merged.csv` | Output merged CSV |

### `summarize_temporal_ablation.py`

Compare temporal ablation experiment results (`EXP_*` directories) against baseline ASR from summary CSVs.

```bash
python sweep/summarize_temporal_ablation.py --csv
python sweep/summarize_temporal_ablation.py --csv --stretch 12 --source G4
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | off | Write `temporal_ablation_summary.csv` |
| `--stretch` | all | Filter to specific stretch value |
| `--source` | all | Filter to specific source (e.g. `G4`, `S13`) |

## Directory Layout

```
sweep/
├── S1_…/ … S10_…/           # Original sweep runs
├── G1_…/ … G12_…/           # Grid search runs
│   ├── tti_uap.pt
│   ├── adv_videos/
│   ├── adv_videos_stretch*_scale*/
│   ├── grid_config.txt
│   └── {model}_summary_metrics.txt
├── EXP_*/                    # Experiment output dirs
├── eval_top_list.txt         # Run names for eval_top
├── postprocess_list.txt      # Source dirs for post-processing
├── quality_summary.csv       # Aggregated quality metrics
├── logs/                     # SLURM log files
└── summary*.csv              # Aggregation outputs
```
