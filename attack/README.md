# attack/ — UAP Generation Pipeline

This directory contains the full pipeline for training, applying, and visualising Temporal Trajectory Injection UAPs.

## Pipeline Overview

```
precompute_accident_temporal.py   →  accident_temporal.pt
                                           ↓
tti_attack.py  ←  normal images    →  tti_uap.pt
                                           ↓
apply_uap.py  ←  clean videos     →  adversarial videos
                                           ↓  (optional)
postprocess_videos.py              →  degraded adversarial videos

apply_static_uap.py  ←  patch image  →  static-baseline adversarial videos
visualise_uap.py  ←  tti_uap.pt      →  per-frame PNGs + grid images
```

## Scripts

### `precompute_accident_temporal.py`

Build accident temporal templates from accident training videos. For each CLIP model, extracts *N*-frame averaged feature trajectories and per-position transition cosine similarities using impact-weighted frame sampling.

| Argument | Default | Description |
|----------|---------|-------------|
| `--video_dir` | *required* | Directory of accident videos (recursive, `.mp4`/`.avi`/`.mov`/`.mkv`/`.wmv`/`.flv`) |
| `--output` | `./accident_temporal.pt` | Output `.pt` file |
| `--N` | `32` | Number of temporal positions |
| `--clip_models` | `ViT-L-14` | Space-separated CLIP architecture names |
| `--clip_pretrained` | `openai` | Default pretrained weights (used when `--clip_pretrained_list` is omitted) |
| `--clip_pretrained_list` | `None` | Per-model pretrained weights (must match length of `--clip_models`) |
| `--impact_weight` | `0.4` | Weight applied to impact-position frames |
| `--max_videos` | `None` | Limit number of videos processed |
| `--device` | `auto` | `cuda` / `cpu` / `auto` |

**Output format:** PyTorch dict with keys `templates`, `transition_sims`, `N`, `clip_models`, `n_videos`, `avg_impact_position`.

---

### `tti_attack.py`

Core training script. Learns an *N*-frame universal perturbation optimised with a CLIP ensemble. Loss components:

- **L_clip** — targeted text alignment (+ optional negative-text repulsion)
- **L_traj** — trajectory matching against accident temporal templates
- **L_trans** — transition cosine-similarity matching

Uses DI-FGSM input diversity and MI-FGSM momentum.

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_dir` | *required* | Normal (non-accident) training images |
| `--output` | `./tti_uap.pt` | Output UAP tensor `(N, C, H, W)` |
| `--epsilon` | `16/255` | L∞ perturbation bound |
| `--alpha` | `2/255` | PGD step size |
| `--N` | `32` | Temporal length of UAP |
| `--clip_models` | *required* | CLIP architectures (space-separated) |
| `--clip_pretrained_list` | *required* | Pretrained weights per model |
| `--target_texts` | *required* | Target text(s) for alignment |
| `--negative_texts` | `None` | Negative text(s) to repel |
| `--lambda_neg` | `0.3` | Weight for negative-text loss |
| `--accident_temporal` | `None` | Path to `accident_temporal.pt` (enables L_traj + L_trans) |
| `--lambda_traj` | `0.3` | Weight for trajectory loss |
| `--lambda_trans` | `0.2` | Weight for transition loss |
| `--di_prob` | `0.5` | DI-FGSM augmentation probability |
| `--di_scale_low` | `0.8` | DI-FGSM minimum scale |
| `--mu` | `0.9` | MI-FGSM momentum decay |
| `--image_size` | `448` | Input resolution |
| `--max_images` | `None` | Limit training images |
| `--epochs` | `10` | Training epochs over image set |
| `--save_every` | `1000` | Checkpoint interval (steps) |
| `--device` | `auto` | `cuda` / `cpu` / `auto` |

---

### `apply_uap.py`

Apply a trained UAP to every video in a directory. The perturbation cycles over frames with configurable stretch (temporal interpolation) and scale factors. Supports temporal ablation modes for analysis.

| Argument | Default | Description |
|----------|---------|-------------|
| `--uap` | *required* | Path to UAP `.pt` file |
| `--video_dir` | *required* | Directory of clean videos (recursive) |
| `--output_dir` | `./adversarial_videos` | Output directory |
| `--codec` | `mp4v` | OpenCV codec fallback |
| `--ext` | `.mp4` | Output extension |
| `--crf` | `23` | FFmpeg CRF (0 = best, 23 = default) |
| `--smooth_sigma` | `0.0` | Temporal Gaussian smoothing sigma |
| `--stretch` | `1` | Temporal stretch factor (interpolation) |
| `--scale` | `1.0` | Perturbation amplitude scale |
| `--epsilon` | `None` | Override ε (auto-detected from UAP after scaling) |
| `--shuffle_seed` | `None` | Shuffle frame order (ablation) |
| `--static_frame` | `None` | Repeat single frame index (ablation) |
| `--reverse` | off | Reverse temporal order (ablation) |

The last three arguments (`--shuffle_seed`, `--static_frame`, `--reverse`) are mutually exclusive temporal ablation modes.

---

### `apply_static_uap.py`

Baseline method: blend a single static image into every frame as `V_adv = (1 − α) V + α · patch`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--patch` | *required* | Path to patch image |
| `--video_dir` | *required* | Directory of clean videos (recursive) |
| `--output_dir` | `./adversarial_videos` | Output directory |
| `--alpha` | `16/255` | Blend strength |
| `--crf` | `23` | FFmpeg CRF |
| `--ext` | `.mp4` | Output extension |

---

### `postprocess_videos.py`

Apply robustness degradations to adversarial videos: per-frame Gaussian noise and/or bilateral filtering. Used to test UAP robustness under post-processing.

| Argument | Default | Description |
|----------|---------|-------------|
| `--video_dir` | *required* | Directory of adversarial videos |
| `--output_dir` | *required* | Output directory |
| `--noise_std` | `0.0` | Gaussian noise standard deviation (0–255 scale) |
| `--bilateral_d` | `0` | Bilateral filter diameter (0 = disabled) |
| `--bilateral_sigma_color` | `75.0` | Bilateral colour sigma |
| `--bilateral_sigma_space` | `75.0` | Bilateral spatial sigma |
| `--crf` | `23` | FFmpeg CRF |
| `--seed` | `42` | Random seed for noise |

At least one of `--noise_std > 0` or `--bilateral_d > 0` is required.

---

### `visualise_uap.py`

Inspection utility: save per-frame PNGs (raw + amplified) and grid images from a UAP `.pt` file.

| Argument | Default | Description |
|----------|---------|-------------|
| `--uap` | *required* | Path to UAP `.pt` file |
| `--output_dir` | `./uap_vis` | Output directory |
| `--amplify` | `10.0` | Amplification factor for visibility |

**Outputs:** `raw/frame_*.png`, `amplified/frame_*.png`, `grid_raw.png`, `grid_amplified.png`.

---

## Shell Scripts

### `run_attack.sh`

SLURM job for the full TTI-UAP pipeline:

1. Pre-compute temporal templates (if `accident_temporal.pt` is missing)
2. Train UAP via `tti_attack.py`
3. Apply UAP via `apply_uap.py`

Submit: `sbatch attack/run_attack.sh`

### `run_exp_static.sh`

SLURM job for the static-patch baseline experiment: applies `apply_static_uap.py` then evaluates on all four VLMs.

Submit: `sbatch attack/run_exp_static.sh`

Both scripts source `config.sh` at the repo root for path configuration.
