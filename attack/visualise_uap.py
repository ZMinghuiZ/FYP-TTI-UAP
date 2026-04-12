#!/usr/bin/env python3
"""
Visualise a TTI-UAP .pt file as images.

Saves:
  - Individual frames as PNGs (one per UAP frame)
  - A grid image showing all N frames side by side
  - Both raw perturbation and amplified (10x) versions for visibility

The raw perturbation values are in [-eps, eps] ≈ [-0.063, +0.063], which is
nearly invisible. The amplified version scales them to fill the full [0, 255]
range so you can actually see the noise patterns.

Usage:
    python visualise_uap.py --uap tti_uap.pt --output_dir ./uap_vis
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Visualise TTI-UAP as images")
    p.add_argument("--uap", type=str, required=True,
                    help="Path to TTI-UAP tensor (.pt file)")
    p.add_argument("--output_dir", type=str, default="./uap_vis",
                    help="Output directory for images (default: ./uap_vis)")
    p.add_argument("--amplify", type=float, default=10.0,
                    help="Amplification factor for visibility (default: 10.0)")
    return p.parse_args()


def tensor_to_image(t, amplify=1.0):
    """
    Convert a perturbation tensor (C, H, W) in [-eps, eps] to a PIL Image.

    The perturbation is shifted to [0, 1] range:
      - Raw (amplify=1):   pixel = delta + 0.5  (grey = zero perturbation)
      - Amplified:         pixel = delta * amplify + 0.5
    """
    img = t.clone().float()
    img = img * amplify + 0.5
    img = img.clamp(0.0, 1.0)
    img = (img * 255).byte()
    # (C, H, W) → (H, W, C)
    img = img.permute(1, 2, 0).numpy()
    return Image.fromarray(img, mode="RGB")


def make_grid(images, cols=8, padding=2, bg_color=128):
    """Arrange a list of PIL Images into a grid."""
    if not images:
        return None
    w, h = images[0].size
    rows = math.ceil(len(images) / cols)

    grid_w = cols * w + (cols + 1) * padding
    grid_h = rows * h + (rows + 1) * padding
    grid = Image.new("RGB", (grid_w, grid_h), (bg_color, bg_color, bg_color))

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = padding + c * (w + padding)
        y = padding + r * (h + padding)
        grid.paste(img, (x, y))

    return grid


def main():
    args = parse_args()

    # Load UAP
    print(f"Loading TTI-UAP from {args.uap} ...")
    delta_N = torch.load(args.uap, map_location="cpu", weights_only=True)
    N = delta_N.shape[0]
    print(f"  Shape: {list(delta_N.shape)}  (N={N})")
    print(f"  Value range: [{delta_N.min().item():.6f}, {delta_N.max().item():.6f}]")
    print()

    out = Path(args.output_dir)
    raw_dir = out / "raw"
    amp_dir = out / "amplified"
    raw_dir.mkdir(parents=True, exist_ok=True)
    amp_dir.mkdir(parents=True, exist_ok=True)

    raw_images = []
    amp_images = []

    for i in range(N):
        frame = delta_N[i]

        # Raw: shift so 0 perturbation = grey (128)
        raw_img = tensor_to_image(frame, amplify=1.0)
        raw_img.save(raw_dir / f"frame_{i:03d}.png")
        raw_images.append(raw_img)

        # Amplified: scale up so patterns are clearly visible
        amp_img = tensor_to_image(frame, amplify=args.amplify)
        amp_img.save(amp_dir / f"frame_{i:03d}.png")
        amp_images.append(amp_img)

    # Save grids
    cols = min(8, N)

    raw_grid = make_grid(raw_images, cols=cols)
    raw_grid.save(out / "grid_raw.png")

    amp_grid = make_grid(amp_images, cols=cols)
    amp_grid.save(out / "grid_amplified.png")

    print(f"Saved {N} individual frames to:")
    print(f"  Raw (true scale):  {raw_dir}/")
    print(f"  Amplified ({args.amplify}x):  {amp_dir}/")
    print()
    print(f"Saved grid overviews to:")
    print(f"  {out / 'grid_raw.png'}")
    print(f"  {out / 'grid_amplified.png'}")


if __name__ == "__main__":
    main()
