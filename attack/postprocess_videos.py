#!/usr/bin/env python3
"""
Post-process adversarial videos with Gaussian noise and bilateral filtering.

Simulates real-world degradation to test whether the adversarial perturbation
survives noise and smoothing.  Operates on already-generated adversarial videos
(from apply_uap.py) so the UAP does not need to be re-applied.

Per-frame pipeline (when both are enabled):
    frame  →  + N(0, noise_std)  →  bilateralFilter  →  output

Usage:
    python postprocess_videos.py \
        --video_dir ./sweep/G3/adv_videos_stretch14_scale1 \
        --output_dir ./sweep/G3/adv_videos_stretch14_scale1_n5_b7 \
        --noise_std 5.0 \
        --bilateral_d 7
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


# ── Video I/O (same helpers as apply_uap.py) ─────────────────────────────


def _ffmpeg_available():
    return shutil.which("ffmpeg") is not None


def _open_ffmpeg_writer(output_path, fps, width, height, crf=23):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _create_cv_writer(output_path, fps, width, height):
    """Fallback OpenCV VideoWriter with codec negotiation."""
    codecs = ["avc1", "mp4v", "XVID", "MJPG"]
    ext_map = {"XVID": ".avi", "MJPG": ".avi", "mp4v": ".mp4", "avc1": ".mp4"}

    for c in codecs:
        fourcc = cv2.VideoWriter_fourcc(*c)
        out = str(Path(output_path).with_suffix(ext_map.get(c, ".avi")))
        writer = cv2.VideoWriter(out, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, Path(out)
        writer.release()
    return None, None


def find_videos(video_dir):
    vdir = Path(video_dir)
    if not vdir.is_dir():
        print(f"Error: '{video_dir}' is not a directory.")
        sys.exit(1)
    videos = sorted(
        p for p in vdir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"  No video files found in {video_dir}")
        print(f"  Looking for: {VIDEO_EXTENSIONS}")
    return videos


# ── Frame processing ──────────────────────────────────────────────────────


def process_frame(frame, noise_std, bilateral_d, sigma_color, sigma_space,
                  rng):
    """Apply Gaussian noise then bilateral filter to a BGR uint8 frame."""
    if noise_std > 0:
        noise = rng.normal(0, noise_std, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if bilateral_d > 0:
        frame = cv2.bilateralFilter(frame, bilateral_d, sigma_color, sigma_space)
    return frame


# ── Per-video processing ─────────────────────────────────────────────────


def postprocess_video(video_path, output_path, noise_std, bilateral_d,
                      sigma_color, sigma_space, crf, rng):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARNING] Cannot open: {video_path.name}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        print(f"  [WARNING] No frames in: {video_path.name}")
        return False

    ffmpeg_proc = None
    writer = None
    actual_output = Path(output_path)

    if _ffmpeg_available():
        actual_output = actual_output.with_suffix(".mp4")
        ffmpeg_proc = _open_ffmpeg_writer(actual_output, fps, width, height,
                                          crf=crf)
    else:
        writer, actual_output = _create_cv_writer(output_path, fps, width,
                                                  height)
        if writer is None:
            cap.release()
            print(f"  [WARNING] No working video codec for: {video_path.name}")
            return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = process_frame(frame, noise_std, bilateral_d,
                              sigma_color, sigma_space, rng)
        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.write(frame.tobytes())
        else:
            writer.write(frame)

    cap.release()

    if ffmpeg_proc is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        if ffmpeg_proc.returncode != 0:
            print(f"  [WARNING] ffmpeg encoding failed for: {video_path.name}")
            return False
    elif writer is not None:
        writer.release()

    if not actual_output.exists() or actual_output.stat().st_size == 0:
        print(f"  [WARNING] Output empty or missing: {actual_output}")
        return False

    return True


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Post-process adversarial videos with noise + bilateral filter")

    p.add_argument("--video_dir", type=str, required=True,
                   help="Directory containing adversarial videos to process")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Output directory for post-processed videos")

    p.add_argument("--noise_std", type=float, default=0.0,
                   help="Gaussian noise std dev in [0,255] scale. "
                        "0 = skip (default: 0)")
    p.add_argument("--bilateral_d", type=int, default=0,
                   help="Bilateral filter diameter. 0 = skip. "
                        "Typical: 5-9 (default: 0)")
    p.add_argument("--bilateral_sigma_color", type=float, default=75.0,
                   help="Bilateral filter sigma in color space (default: 75)")
    p.add_argument("--bilateral_sigma_space", type=float, default=75.0,
                   help="Bilateral filter sigma in coordinate space (default: 75)")

    p.add_argument("--crf", type=int, default=23,
                   help="H.264 CRF quality (0=best, 23=default, 51=worst)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible noise (default: 42)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.noise_std <= 0 and args.bilateral_d <= 0:
        print("ERROR: At least one of --noise_std or --bilateral_d must be > 0.")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)

    # ── Summary ──
    ops = []
    if args.noise_std > 0:
        ops.append(f"Gaussian noise (std={args.noise_std:.1f})")
    if args.bilateral_d > 0:
        ops.append(f"bilateral filter (d={args.bilateral_d}, "
                   f"σ_color={args.bilateral_sigma_color}, "
                   f"σ_space={args.bilateral_sigma_space})")
    print(f"Post-processing pipeline: {' → '.join(ops)}")
    print(f"  CRF  : {args.crf}")
    print(f"  Seed : {args.seed}")
    print()

    # ── Find videos ──
    videos = find_videos(args.video_dir)
    print(f"Found {len(videos)} video(s) in {args.video_dir}")
    if not videos:
        print("Nothing to process. Exiting.")
        sys.exit(0)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if _ffmpeg_available():
        print(f"Using ffmpeg for H.264 encoding (CRF={args.crf}).\n")
    else:
        print("ffmpeg not found, falling back to OpenCV VideoWriter.\n")

    # ── Process ──
    success = 0
    for video_path in tqdm(videos, desc="Post-processing", unit="video"):
        out_path = out_dir / (video_path.stem + ".mp4")
        ok = postprocess_video(
            video_path, out_path,
            noise_std=args.noise_std,
            bilateral_d=args.bilateral_d,
            sigma_color=args.bilateral_sigma_color,
            sigma_space=args.bilateral_sigma_space,
            crf=args.crf,
            rng=rng,
        )
        if ok:
            success += 1

    print(f"\nDone. {success}/{len(videos)} videos saved to {out_dir}")


if __name__ == "__main__":
    main()
