#!/usr/bin/env python3
"""
Summarize PSNR/SSIM results and join with ASR data into a single CSV.

Reads per-video quality_metrics.csv files from adv_videos_stretch*_scale*
directories, aggregates them into per-directory summaries, and merges with
ASR data.  Works even if quality_summary.csv was never generated (e.g.
because the sweep job timed out).

Usage:
    python sweep/summarize_quality.py --run_dir sweep/G4_a4-near-s8-high
    python sweep/summarize_quality.py --run_dir sweep/G4_a4-near-s8-high --output sweep/quality_asr_merged.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

_DIR_RE = re.compile(r"adv_videos_stretch(\d+)_scale([\d.]+)")


def parse_stretch_scale(dirname: str):
    """Extract (stretch, scale) from a directory name."""
    m = _DIR_RE.search(dirname)
    if m is None:
        return None, None
    return int(m.group(1)), float(m.group(2))


def discover_quality_csvs(run_dir: Path) -> list[Path]:
    """Find all quality_metrics.csv under adv_videos_stretch* directories."""
    found = []
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir() or not _DIR_RE.search(d.name):
            continue
        qm = d / "quality_metrics.csv"
        if qm.exists():
            found.append(qm)
    return found


def aggregate_directory(qm_path: Path) -> dict:
    """Aggregate a per-video quality_metrics.csv into a directory summary."""
    df = pd.read_csv(qm_path)
    adv_dir = qm_path.parent
    stretch, scale = parse_stretch_scale(adv_dir.name)
    return {
        "directory": str(adv_dir),
        "stretch": stretch,
        "scale": scale,
        "num_videos": len(df),
        "psnr_mean": float(np.mean(df["psnr_mean"])),
        "psnr_std": float(np.std(df["psnr_mean"])),
        "ssim_mean": float(np.mean(df["ssim_mean"])),
        "ssim_std": float(np.std(df["ssim_mean"])),
    }


def load_asr(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["stretch"] = df["stretch"].astype(int)
    df["scale"] = df["scale"].astype(float)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-video PSNR/SSIM and merge with ASR data")
    parser.add_argument(
        "--run_dir", type=str,
        default=str(SCRIPT_DIR / "G4_a4-near-s8-high"),
        help="Run directory containing adv_videos_stretch*_scale* subdirs")
    parser.add_argument(
        "--asr", type=str,
        default=str(SCRIPT_DIR / "summary_G4_stretch_scale.csv"),
        help="Path to ASR summary CSV")
    parser.add_argument(
        "--output", type=str,
        default=str(SCRIPT_DIR / "quality_asr_merged.csv"),
        help="Output path for merged CSV")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    csvs = discover_quality_csvs(run_dir)
    print(f"Found {len(csvs)} quality_metrics.csv files under {run_dir}")

    if not csvs:
        print("Nothing to summarize.")
        return

    rows = []
    for qm in csvs:
        row = aggregate_directory(qm)
        print(f"  {qm.parent.name}: {row['num_videos']} videos, "
              f"PSNR={row['psnr_mean']:.2f}±{row['psnr_std']:.2f}, "
              f"SSIM={row['ssim_mean']:.4f}±{row['ssim_std']:.4f}")
        rows.append(row)

    quality = pd.DataFrame(rows)
    quality.sort_values(["stretch", "scale"], inplace=True)
    quality.reset_index(drop=True, inplace=True)

    asr_path = Path(args.asr)
    if asr_path.exists():
        asr = load_asr(asr_path)
        merged = quality.merge(
            asr[["stretch", "scale", "internvl_asr", "qwen_asr",
                 "llava_onevision_asr", "videollama3_asr", "mean_asr"]],
            on=["stretch", "scale"],
            how="left",
        )
        print(f"\nJoined with {len(asr)} ASR rows from {asr_path}")
    else:
        merged = quality.copy()
        print(f"\nASR CSV not found ({asr_path}), quality only.")

    out = Path(args.output)
    merged.to_csv(out, index=False)
    print(f"Saved {len(merged)} rows to {out}")


if __name__ == "__main__":
    main()
