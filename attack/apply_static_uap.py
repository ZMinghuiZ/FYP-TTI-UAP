#!/usr/bin/env python3
"""
Apply a static (image-based) UAP to videos via alpha blending.

Given a UAP image (e.g. a JPEG), this script:
  1. Reads each source video frame-by-frame.
  2. Blends the UAP into every frame:  V_adv^t = (1-α)·V^t + α·UAP
  3. Writes the adversarial video to the output directory.

Usage:
    python apply_static_uap.py \
        --patch static/000000.jpg \
        --video_dir /path/to/videos \
        --output_dir ./adversarial_videos \
        --alpha 0.0627
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


def parse_args():
    p = argparse.ArgumentParser(description="Apply a static UAP image to videos")

    p.add_argument("--patch", type=str, required=True,
                    help="Path to the UAP image file (e.g. static/000000.jpg)")
    p.add_argument("--video_dir", type=str, required=True,
                    help="Directory containing source videos")
    p.add_argument("--output_dir", type=str, default="./adversarial_videos",
                    help="Output directory for adversarial videos")
    p.add_argument("--alpha", type=float, default=16 / 255,
                    help="Blending weight for the UAP (default: 16/255 ≈ 0.0627)")
    p.add_argument("--crf", type=int, default=23,
                    help="H.264 CRF quality (0=best, 23=default, 51=worst)")
    p.add_argument("--ext", type=str, default=".mp4",
                    help="Output file extension (default: .mp4)")

    return p.parse_args()


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
        all_files = [f for f in vdir.rglob("*") if f.is_file()]
        print(f"  Directory contents ({len(all_files)} files):")
        extensions = {}
        for f in all_files:
            ext = f.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        for ext, count in sorted(extensions.items()):
            print(f"    {ext or '(no extension)'}: {count} file(s)")
        print(f"  Looking for: {VIDEO_EXTENSIONS}")

    return videos


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


def apply_static_uap_to_video(video_path, patch_bgr, alpha, output_path, crf=23):
    """
    Read a video, alpha-blend the static UAP into every frame, write result.

    Blending: frame_out = (1 - alpha) * frame_in + alpha * patch
    """
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

    patch_resized = cv2.resize(patch_bgr, (width, height),
                                interpolation=cv2.INTER_LINEAR)
    patch_f32 = patch_resized.astype(np.float32)

    actual_output = Path(output_path)

    if _ffmpeg_available():
        actual_output = actual_output.with_suffix(".mp4")
        ffmpeg_proc = _open_ffmpeg_writer(actual_output, fps, width, height, crf=crf)
        writer = None
    else:
        ffmpeg_proc = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(actual_output), fourcc, fps, (width, height))
        if not writer.isOpened():
            cap.release()
            print(f"  [WARNING] Cannot create writer for: {video_path.name}")
            return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_f32 = frame.astype(np.float32)
        blended = cv2.addWeighted(frame_f32, 1.0 - alpha, patch_f32, alpha, 0.0)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        if ffmpeg_proc is not None:
            ffmpeg_proc.stdin.write(blended.tobytes())
        else:
            writer.write(blended)

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
        print(f"  [WARNING] Output file is empty or missing: {actual_output}")
        return False

    return True


def main():
    args = parse_args()

    print(f"Loading static UAP from {args.patch} ...")
    patch_bgr = cv2.imread(args.patch)
    if patch_bgr is None:
        print(f"Error: Could not read patch image from {args.patch}")
        sys.exit(1)

    h, w = patch_bgr.shape[:2]
    print(f"  Patch size: {w}x{h}")
    print(f"  Alpha (blend weight): {args.alpha:.6f}  ({args.alpha * 255:.1f}/255)")
    print(f"  CRF: {args.crf}")
    print()

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

    success = 0
    for video_path in tqdm(videos, desc="Applying static UAP", unit="video"):
        out_name = video_path.stem + "_adv" + args.ext
        out_path = out_dir / out_name

        ok = apply_static_uap_to_video(
            video_path, patch_bgr, args.alpha, out_path, crf=args.crf,
        )
        if ok:
            success += 1

    print(f"\nDone. {success}/{len(videos)} adversarial videos saved to {out_dir}")


if __name__ == "__main__":
    main()
