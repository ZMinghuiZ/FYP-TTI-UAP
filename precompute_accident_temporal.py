#!/usr/bin/env python3
"""
Pre-compute accident temporal templates from accident videos.

Extracts CLIP features from accident video frames and produces:
  1. Per-model temporal templates: averaged N-frame feature trajectories
     for each CLIP model in the training ensemble.  Shape (N, D).
  2. Per-model transition similarities: average cosine similarity between
     consecutive frames at each position.  Shape (N-1,).

Frame sampling uses impact-weighted density: a fast pixel-difference scan
detects the "impact" frame, then samples more densely around it so the
template captures the sharp feature shift at the collision point.

Usage:
    python precompute_accident_temporal.py \\
        --video_dir /path/to/accident/videos \\
        --output accident_temporal.pt \\
        --N 32 \\
        --clip_models ViT-L-14 EVA02-L-14 ViT-SO400M-14-SigLIP \\
        --clip_pretrained_list openai merged2b_s4b_b131k webli
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-compute accident temporal templates from videos"
    )
    p.add_argument("--video_dir", type=str, required=True,
                   help="Directory containing accident videos")
    p.add_argument("--output", type=str, default="./accident_temporal.pt",
                   help="Output .pt path")
    p.add_argument("--N", type=int, default=32,
                   help="Number of frames to sample per video (default: 32)")
    p.add_argument("--clip_models", type=str, nargs="+",
                   default=["ViT-L-14"],
                   help="OpenCLIP model name(s)")
    p.add_argument("--clip_pretrained_list", type=str, nargs="+",
                   default=None,
                   help="Per-model pretrained weights, 1:1 with --clip_models")
    p.add_argument("--clip_pretrained", type=str, default="openai",
                   help="Default pretrained when list not set")
    p.add_argument("--impact_weight", type=float, default=0.4,
                   help="Fraction of N frames concentrated near impact "
                        "(default: 0.4)")
    p.add_argument("--max_videos", type=int, default=None,
                   help="Cap number of videos processed")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: cuda | cpu | mps | auto")
    return p.parse_args()


def get_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_videos(video_dir):
    vdir = Path(video_dir)
    if not vdir.is_dir():
        print(f"Error: '{video_dir}' is not a directory.")
        sys.exit(1)
    return sorted(
        p for p in vdir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _detect_impact_frame(video_path, scan_count):
    """Fast pixel-level scan to find the frame with maximum visual change."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return total, 0

    scan_count = min(total, scan_count)
    scan_indices = np.linspace(0, total - 1, scan_count, dtype=int)
    prev_thumb = None
    diffs = []

    for idx in scan_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        thumb = cv2.resize(frame, (64, 64)).astype(np.float32) / 255.0
        if prev_thumb is not None:
            diffs.append(np.mean(np.abs(thumb - prev_thumb)))
        prev_thumb = thumb

    cap.release()
    if not diffs:
        return total, 0

    impact_scan_idx = int(np.argmax(diffs))
    impact_video_idx = int(scan_indices[impact_scan_idx])
    return total, impact_video_idx


def sample_n_frames_impact_weighted(video_path, n_frames, impact_weight=0.4):
    """Sample n_frames with higher density around the detected impact point.

    Returns (frames, impact_position) where impact_position is the
    fractional position [0, 1] of the impact within the selected frames.
    """
    total, impact_video_idx = _detect_impact_frame(
        video_path, scan_count=max(4 * n_frames, 64))
    if total is None or total < n_frames:
        return None, None

    n_impact = max(1, int(n_frames * impact_weight))
    n_uniform = n_frames - n_impact

    uniform_idx = np.linspace(0, total - 1, n_uniform, dtype=int)

    radius = max(1, int(total * 0.10))
    impact_start = max(0, impact_video_idx - radius)
    impact_end = min(total - 1, impact_video_idx + radius)
    dense_idx = np.linspace(impact_start, impact_end, n_impact, dtype=int)

    combined = np.unique(np.concatenate([uniform_idx, dense_idx]))
    if len(combined) > n_frames:
        sel = np.linspace(0, len(combined) - 1, n_frames, dtype=int)
        combined = combined[sel]
    elif len(combined) < n_frames:
        extra = np.linspace(0, total - 1, n_frames * 2, dtype=int)
        combined = np.unique(np.concatenate([combined, extra]))
        combined = combined[:n_frames] if len(combined) >= n_frames else combined
    combined = np.sort(combined)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None

    frames = []
    for idx in combined:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None, None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb.astype(np.float32) / 255.0)
    cap.release()

    distances = np.abs(combined - impact_video_idx)
    impact_local = int(np.argmin(distances))
    impact_frac = impact_local / max(len(combined) - 1, 1)
    return frames, impact_frac


def encode_frame(frame_np, clip_model, clip_mean, clip_std,
                 clip_input_size, device):
    """Encode a single numpy RGB [0,1] frame -> (1, D) L2-normalised."""
    tensor = torch.from_numpy(frame_np).permute(2, 0, 1).unsqueeze(0).to(device)
    tensor = F.interpolate(
        tensor, size=(clip_input_size, clip_input_size),
        mode="bilinear", align_corners=False,
    )
    tensor = (tensor - clip_mean) / clip_std
    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat


def main():
    args = parse_args()
    device = get_device(args.device)
    N = args.N

    if args.clip_pretrained_list:
        if len(args.clip_pretrained_list) != len(args.clip_models):
            print("ERROR: --clip_pretrained_list must match --clip_models length",
                  file=sys.stderr)
            sys.exit(1)
        clip_pretrained_list = args.clip_pretrained_list
    else:
        clip_pretrained_list = [args.clip_pretrained] * len(args.clip_models)

    print("=" * 60)
    print("Pre-compute Accident Temporal Templates (TTI-UAP)")
    print("=" * 60)
    model_strs = [f"{m}({p})" for m, p in
                  zip(args.clip_models, clip_pretrained_list)]
    print(f"  CLIP models   : {', '.join(model_strs)}")
    print(f"  N frames      : {N}")
    print(f"  Impact weight : {args.impact_weight}")
    print(f"  Device        : {device}")
    print()

    # ── Load CLIP models ─────────────────────────────────────────────
    import open_clip

    clip_models_info = []
    for cname, cpretrained in zip(args.clip_models, clip_pretrained_list):
        print(f"Loading CLIP {cname} ({cpretrained}) ...")
        model_obj, _, preprocess = open_clip.create_model_and_transforms(
            cname, pretrained=cpretrained, device=device,
        )
        model_obj.eval()
        for p in model_obj.parameters():
            p.requires_grad = False

        input_size = preprocess.transforms[0].size
        if isinstance(input_size, (tuple, list)):
            input_size = input_size[0]

        mean = torch.tensor(
            getattr(open_clip, "OPENAI_DATASET_MEAN",
                    (0.48145466, 0.4578275, 0.40821073)),
            device=device,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            getattr(open_clip, "OPENAI_DATASET_STD",
                    (0.26862954, 0.26130258, 0.27577711)),
            device=device,
        ).view(1, 3, 1, 1)

        clip_models_info.append({
            "key": f"{cname}:{cpretrained}",
            "model": model_obj,
            "input_size": input_size,
            "mean": mean,
            "std": std,
        })
        print(f"  {cname}:{cpretrained}: input_size={input_size}")

    print(f"{len(clip_models_info)} CLIP model(s) loaded.\n")

    # ── Find videos ──────────────────────────────────────────────────
    videos = find_videos(args.video_dir)
    if args.max_videos and len(videos) > args.max_videos:
        idx = np.linspace(0, len(videos) - 1, args.max_videos, dtype=int)
        videos = [videos[i] for i in idx]
    print(f"Found {len(videos)} accident video(s) in {args.video_dir}")
    if not videos:
        print("No videos found. Exiting.")
        sys.exit(1)
    print()

    # ── Extract features ─────────────────────────────────────────────
    all_features_per_model = {info["key"]: [] for info in clip_models_info}
    impact_positions = []

    for vpath in tqdm(videos, desc="Extracting features", unit="video"):
        frames, impact_frac = sample_n_frames_impact_weighted(
            vpath, N, impact_weight=args.impact_weight)
        if frames is None:
            tqdm.write(f"  [SKIP] {vpath.name} (too few frames or unreadable)")
            continue

        impact_positions.append(impact_frac)

        for info in clip_models_info:
            frame_features = []
            for frame_np in frames:
                feat = encode_frame(
                    frame_np, info["model"], info["mean"], info["std"],
                    info["input_size"], device,
                )
                frame_features.append(feat.squeeze(0))
            all_features_per_model[info["key"]].append(
                torch.stack(frame_features))

    primary_key = clip_models_info[0]["key"]
    n_videos = len(all_features_per_model[primary_key])
    if n_videos == 0:
        print("No valid videos processed. Exiting.")
        sys.exit(1)

    avg_impact = float(np.mean(impact_positions)) if impact_positions else 0.5
    print(f"\nProcessed {n_videos} video(s) successfully.")
    print(f"  Average impact position: {avg_impact:.2%} "
          f"(frame ~{int(avg_impact * (N - 1))}/{N - 1})")

    # ── Build per-model templates and transition similarities ────────
    templates = {}
    transition_sims = {}

    for info in clip_models_info:
        key = info["key"]
        model_feats = all_features_per_model[key]
        stacked = torch.stack(model_feats)        # (V, N, D)

        # Trajectory template: averaged feature per position
        template = stacked.mean(dim=0)             # (N, D)
        template = template / template.norm(dim=-1, keepdim=True)
        templates[key] = template.cpu()

        # Transition similarities: mean cos(frame_n, frame_{n+1}) per position
        trans = torch.zeros(N - 1)
        for v_feats in model_feats:                # each is (N, D)
            for n in range(N - 1):
                sim = F.cosine_similarity(
                    v_feats[n:n + 1], v_feats[n + 1:n + 2]).item()
                trans[n] += sim
        trans /= len(model_feats)
        transition_sims[key] = trans.cpu()

        feat_dim = model_feats[0].shape[1]
        print(f"\n  {key}: template ({N}, {feat_dim})")

        # Show inter-position similarities of the template itself
        if N > 1:
            t_sims = [
                F.cosine_similarity(
                    template[i:i + 1], template[i + 1:i + 2]).item()
                for i in range(min(N - 1, 5))
            ]
            print(f"    template inter-pos sims (first 5): "
                  f"{[f'{s:.4f}' for s in t_sims]}")

        # Show transition sim pattern (highlights the impact dip)
        if N > 1:
            t_min_idx = int(trans.argmin())
            t_min_val = trans[t_min_idx].item()
            t_max_val = trans.max().item()
            t_mean_val = trans.mean().item()
            print(f"    transition sims: mean={t_mean_val:.4f}, "
                  f"min={t_min_val:.4f} (pos {t_min_idx}), "
                  f"max={t_max_val:.4f}")

    # ── Save ─────────────────────────────────────────────────────────
    output_data = {
        "templates": templates,
        "transition_sims": transition_sims,
        "N": N,
        "clip_models": [f"{m}:{p}" for m, p in
                        zip(args.clip_models, clip_pretrained_list)],
        "n_videos": n_videos,
        "avg_impact_position": avg_impact,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, args.output)

    print(f"\nSaved to {args.output}")
    print(f"  templates      : {list(templates.keys())}")
    print(f"  transition_sims: {list(transition_sims.keys())}")
    print(f"  From {n_videos} video(s)")


if __name__ == "__main__":
    main()
