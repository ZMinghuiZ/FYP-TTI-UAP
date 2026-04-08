#!/usr/bin/env python3
"""
Compute PSNR and SSIM between clean and adversarial video pairs.

For each adversarial video, finds the corresponding clean video by stripping
the '_adv' suffix, then computes per-frame PSNR and SSIM.  Supports
multiprocessing for parallel video processing within each directory.

Usage:
    python compute_quality.py \\
        --clean_dir /path/to/clean \\
        --adv_dir sweep/G10/adv_videos_stretch14_scale1 \\
        --workers 4

    # 10-30x faster: sample every 10th frame (negligible accuracy loss for UAPs)
    python compute_quality.py \\
        --clean_dir /path/to/clean \\
        --adv_dir dir1 dir2 dir3 \\
        --workers 8 --sample_every 10 --summary sweep/quality_summary.csv
"""

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

_SAMPLE_EVERY = 1  # module-level default; overwritten by CLI arg


# ---------------------------------------------------------------------------
# Clean-video lookup
# ---------------------------------------------------------------------------

def build_clean_lookup(clean_dir):
    """Build a {stem: path} mapping for all video files under *clean_dir*."""
    lookup = {}
    for p in clean_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            lookup[p.stem] = p
    return lookup


def find_clean_video(adv_path, clean_lookup):
    """Resolve the clean counterpart of an adversarial video."""
    stem = adv_path.stem
    clean_stem = stem[:-4] if stem.endswith("_adv") else stem
    return clean_lookup.get(clean_stem)


# ---------------------------------------------------------------------------
# Per-video quality computation
# ---------------------------------------------------------------------------

def _compute_video_metrics(args):
    """Worker function: compute mean/std PSNR & SSIM for one video pair."""
    clean_path, adv_path = args
    sample_every = _SAMPLE_EVERY

    clean_cap = cv2.VideoCapture(str(clean_path))
    adv_cap = cv2.VideoCapture(str(adv_path))

    if not clean_cap.isOpened():
        return {"error": f"Cannot open clean video: {clean_path}"}
    if not adv_cap.isOpened():
        clean_cap.release()
        return {"error": f"Cannot open adversarial video: {adv_path}"}

    total_frames = int(adv_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    psnr_vals = []
    ssim_vals = []
    frame_idx = 0

    while True:
        ok_c, frame_c = clean_cap.read()
        ok_a, frame_a = adv_cap.read()
        if not ok_c or not ok_a:
            break

        if frame_idx % sample_every == 0:
            if frame_c.shape != frame_a.shape:
                frame_c = cv2.resize(
                    frame_c, (frame_a.shape[1], frame_a.shape[0])
                )

            psnr_vals.append(cv2.PSNR(frame_c, frame_a))
            ssim_vals.append(
                structural_similarity(
                    frame_c, frame_a, channel_axis=2, data_range=255
                )
            )

        frame_idx += 1

    clean_cap.release()
    adv_cap.release()

    n_sampled = len(psnr_vals)
    if n_sampled == 0:
        return {"error": f"No frames read from {adv_path.name}"}

    return {
        "filename": adv_path.name,
        "psnr_mean": float(np.mean(psnr_vals)),
        "psnr_std": float(np.std(psnr_vals)),
        "ssim_mean": float(np.mean(ssim_vals)),
        "ssim_std": float(np.std(ssim_vals)),
        "num_frames": n_sampled,
        "total_frames": frame_idx,
    }


# ---------------------------------------------------------------------------
# Directory processing
# ---------------------------------------------------------------------------

def process_directory(adv_dir, clean_lookup, workers):
    """Compute quality metrics for every matched pair in *adv_dir*."""
    adv_videos = sorted(
        p for p in adv_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not adv_videos:
        print(f"  No video files found in {adv_dir}")
        return []

    pairs = []
    skipped = 0
    for adv_path in adv_videos:
        clean_path = find_clean_video(adv_path, clean_lookup)
        if clean_path is None:
            skipped += 1
            continue
        pairs.append((clean_path, adv_path))

    if skipped:
        print(f"  Skipped {skipped} video(s) with no clean match")

    if not pairs:
        print(f"  No matched video pairs in {adv_dir}")
        return []

    n_workers = min(workers, len(pairs))
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(_compute_video_metrics, pairs)
    else:
        results = [_compute_video_metrics(p) for p in pairs]

    successes = []
    for r in results:
        if "error" in r:
            print(f"  WARNING: {r['error']}")
        else:
            successes.append(r)

    return successes


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_PER_VIDEO_FIELDS = [
    "filename", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std",
    "num_frames", "total_frames",
]
_SUMMARY_FIELDS = [
    "directory", "num_videos", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std"
]


def write_per_video_csv(results, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PER_VIDEO_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def make_directory_summary(results, adv_dir):
    if not results:
        return None
    psnr_means = [r["psnr_mean"] for r in results]
    ssim_means = [r["ssim_mean"] for r in results]
    return {
        "directory": str(adv_dir),
        "num_videos": len(results),
        "psnr_mean": float(np.mean(psnr_means)),
        "psnr_std": float(np.std(psnr_means)),
        "ssim_mean": float(np.mean(ssim_means)),
        "ssim_std": float(np.std(ssim_means)),
    }


def append_summary_row(summary, csv_path):
    csv_path = Path(csv_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global _SAMPLE_EVERY

    parser = argparse.ArgumentParser(
        description="Compute PSNR and SSIM between clean and adversarial "
                    "video pairs")
    parser.add_argument("--clean_dir", type=str, required=True,
                        help="Path to clean evaluation videos")
    parser.add_argument("--adv_dir", type=str, nargs="+", required=True,
                        help="One or more adversarial video directories")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers per directory (default: 4)")
    parser.add_argument("--sample_every", type=int, default=1,
                        help="Compute metrics on every Nth frame (default: 1 "
                             "= all frames).  10 gives ~10x speedup with "
                             "negligible accuracy change for UAPs.")
    parser.add_argument("--summary", type=str, default=None,
                        help="Append per-directory summary to this CSV")
    args = parser.parse_args()

    _SAMPLE_EVERY = max(1, args.sample_every)

    clean_dir = Path(args.clean_dir)
    if not clean_dir.is_dir():
        print(f"ERROR: Clean directory not found: {clean_dir}")
        sys.exit(1)

    print(f"Clean videos  : {clean_dir}")
    print(f"Workers       : {args.workers}")
    print(f"Sample every  : {_SAMPLE_EVERY} frame(s)")
    print(f"Building clean-video lookup ...")
    clean_lookup = build_clean_lookup(clean_dir)
    print(f"  {len(clean_lookup)} clean video(s) indexed.")
    print(flush=True)

    t_start = time.time()

    for adv_dir_str in args.adv_dir:
        adv_dir = Path(adv_dir_str)
        if not adv_dir.is_dir():
            print(f"WARNING: Skipping missing directory: {adv_dir}")
            continue

        t_dir = time.time()
        print(f"Processing: {adv_dir}")
        results = process_directory(adv_dir, clean_lookup, args.workers)

        if not results:
            print(f"  No results for {adv_dir}\n")
            continue

        per_video_csv = adv_dir / "quality_metrics.csv"
        write_per_video_csv(results, per_video_csv)
        print(f"  Per-video CSV : {per_video_csv}")

        summary = make_directory_summary(results, adv_dir)
        elapsed_dir = time.time() - t_dir
        print(f"  Videos : {summary['num_videos']}  "
              f"PSNR : {summary['psnr_mean']:.2f} +/- "
              f"{summary['psnr_std']:.2f} dB  "
              f"SSIM : {summary['ssim_mean']:.4f} +/- "
              f"{summary['ssim_std']:.4f}  "
              f"({elapsed_dir:.1f}s)")

        if args.summary:
            append_summary_row(summary, args.summary)
            print(f"  Summary row -> {args.summary}")

        print(flush=True)

    elapsed = time.time() - t_start
    print(f"Done. Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
