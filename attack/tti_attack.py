#!/usr/bin/env python3
"""
Temporal Trajectory Injection UAP (TTI-UAP) for Video-LLMs.

Generates a universal N-frame perturbation that injects accident temporal
signatures into normal driving videos, fooling Video-LLMs (InternVL3,
Qwen3-VL) into reporting accidents during Yes/No Video QA.

Three complementary losses optimised jointly:
    L_clip  : per-frame CLIP targeted semantic shift
    L_traj  : trajectory matching against accident temporal templates
    L_trans : inter-frame transition pattern matching

All N frames are updated per training image via batched CLIP forward
passes, giving each frame 32x more gradient signal than the previous
single-frame-per-image approach.

Usage:
    python tti_attack.py \\
        --image_dir /path/to/normal/images \\
        --clip_models ViT-L-14 EVA02-L-14 ViT-SO400M-14-SigLIP \\
        --clip_pretrained_list openai merged2b_s4b_b131k webli \\
        --target_texts "Yes, there is a road accident" \\
        --negative_texts "normal traffic flow" \\
        --accident_temporal ./accident_temporal.pt \\
        --output tti_uap.pt
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

try:
    import open_clip
    HAS_OPEN_CLIP = True
except ImportError:
    HAS_OPEN_CLIP = False


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ── Dataset ───────────────────────────────────────────────────────────────


class ImageFolder(torch.utils.data.Dataset):
    """Load images from a (possibly nested) directory."""

    def __init__(self, root_dir, transform=None, max_images=None):
        self.transform = transform
        self.paths = sorted(
            p for p in Path(root_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        )
        if max_images and len(self.paths) > max_images:
            idx = np.linspace(0, len(self.paths) - 1, max_images, dtype=int)
            self.paths = [self.paths[i] for i in idx]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# ── CLIP Feature Extractor ────────────────────────────────────────────────


class CLIPFeatureExtractor:
    """
    Extract L2-normalised global features from an OpenCLIP vision encoder.

    Inputs are expected in [0, 1] pixel space. CLIP-specific normalisation
    and resizing are applied internally.
    """

    def __init__(self, model_name, pretrained, device):
        if not HAS_OPEN_CLIP:
            raise ImportError("open_clip_torch is required. "
                              "Install with: pip install open_clip_torch")
        self.device = device
        self.model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device,
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        input_size = preprocess.transforms[0].size
        self.input_size = (input_size[0]
                           if isinstance(input_size, (tuple, list))
                           else input_size)

        self.mean = torch.tensor(
            getattr(open_clip, "OPENAI_DATASET_MEAN",
                    (0.48145466, 0.4578275, 0.40821073)),
            device=device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            getattr(open_clip, "OPENAI_DATASET_STD",
                    (0.26862954, 0.26130258, 0.27577711)),
            device=device,
        ).view(1, 3, 1, 1)
        self.model_name = model_name
        self.key = f"{model_name}:{pretrained}"

    def __call__(self, x):
        """Encode images (B, 3, H, W) in [0, 1] -> (B, D) L2-normalised."""
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode="bilinear", align_corners=False,
            )
        x = (x - self.mean) / self.std
        feat = self.model.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True)

    def encode_target_texts(self, texts):
        """Encode a list of text descriptions -> (1, D) averaged, L2-normed."""
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        avg = text_feat.mean(dim=0, keepdim=True)
        return avg / avg.norm(dim=-1, keepdim=True)

    def remove(self):
        pass


# ── Input Diversity ───────────────────────────────────────────────────────


def input_diversity(x, image_size, prob=0.5, scale_low=0.8):
    """DI-FGSM: random resize-and-pad (Xie et al. 2019)."""
    if prob <= 0 or torch.rand(1).item() > prob:
        return x
    new_size = int(image_size * (scale_low + torch.rand(1).item() * (1.0 - scale_low)))
    x_resized = F.interpolate(
        x, size=(new_size, new_size), mode="bilinear", align_corners=False
    )
    pad_h = image_size - new_size
    pad_w = image_size - new_size
    top = torch.randint(0, pad_h + 1, (1,)).item()
    left = torch.randint(0, pad_w + 1, (1,)).item()
    return F.pad(x_resized, (left, pad_w - left, top, pad_h - top))


# ── Loss Functions ────────────────────────────────────────────────────────


def clip_targeted_loss(feat, text_target, text_negative=None, lambda_neg=0.3):
    """
    Per-frame CLIP targeted loss on pre-extracted features.

        L = -cos(feat, text_target) + lambda_neg * cos(feat, text_negative)

    feat: (N, D), text_target: (1, D), text_negative: (1, D) or None.
    """
    loss = -F.cosine_similarity(feat, text_target).mean()
    if text_negative is not None:
        loss = loss + lambda_neg * F.cosine_similarity(
            feat, text_negative).mean()
    return loss


def temporal_trajectory_loss(feat, template):
    """
    Push each frame's features toward the accident temporal template.

        L_traj = (1/N) * sum_n [ 1 - cos(feat_n, template_n) ]

    feat: (N, D), template: (N, D).
    """
    return (1.0 - F.cosine_similarity(feat, template)).mean()


def temporal_transition_loss(feat, target_trans):
    """
    Match inter-frame similarity pattern to accident transition signature.

        L_trans = (1/(N-1)) * sum_n [ (cos(feat_n, feat_{n+1}) - target_n)^2 ]

    feat: (N, D), target_trans: (N-1,) target cosine similarities.
    """
    current_sims = F.cosine_similarity(feat[:-1], feat[1:])
    return ((current_sims - target_trans) ** 2).mean()


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate TTI-UAP for video-LLM attack")

    p.add_argument("--image_dir", type=str, required=True,
                   help="Directory of normal (non-accident) images")
    p.add_argument("--output", type=str, default="./tti_uap.pt",
                   help="Output path for UAP tensor (.pt)")

    p.add_argument("--epsilon", type=float, default=16 / 255,
                   help="L-inf perturbation budget (default: 16/255)")
    p.add_argument("--alpha", type=float, default=2 / 255,
                   help="PGD step size (default: 2/255)")
    p.add_argument("--N", type=int, default=32,
                   help="Number of UAP frames (default: 32)")

    p.add_argument("--clip_models", type=str, nargs="+", required=True,
                   help="OpenCLIP model name(s) for surrogate ensemble")
    p.add_argument("--clip_pretrained_list", type=str, nargs="+", required=True,
                   help="Per-model pretrained weights, 1:1 with --clip_models")
    p.add_argument("--target_texts", type=str, nargs="+", required=True,
                   help="Target text descriptions (averaged)")
    p.add_argument("--negative_texts", type=str, nargs="+", default=None,
                   help="Negative text descriptions to push away from")
    p.add_argument("--lambda_neg", type=float, default=0.3,
                   help="Weight for negative text loss (default: 0.3)")

    # Temporal losses
    p.add_argument("--accident_temporal", type=str, default=None,
                   help="Path to pre-computed accident temporal templates "
                        "(.pt from precompute_accident_temporal.py)")
    p.add_argument("--lambda_traj", type=float, default=0.3,
                   help="Weight for trajectory matching loss (default: 0.3)")
    p.add_argument("--lambda_trans", type=float, default=0.2,
                   help="Weight for transition pattern loss (default: 0.2)")

    # Transferability
    p.add_argument("--di_prob", type=float, default=0.5,
                   help="DI-FGSM input diversity probability (default: 0.5)")
    p.add_argument("--di_scale_low", type=float, default=0.8,
                   help="Minimum scale for DI-FGSM resize (default: 0.8)")
    p.add_argument("--mu", type=float, default=0.9,
                   help="MI-FGSM momentum decay (default: 0.9)")

    p.add_argument("--image_size", type=int, default=448,
                   help="Training image size (default: 448)")
    p.add_argument("--max_images", type=int, default=None,
                   help="Cap number of source images")
    p.add_argument("--epochs", type=int, default=10,
                   help="Passes over the image dataset (default: 10)")
    p.add_argument("--save_every", type=int, default=1000,
                   help="Save checkpoint every N steps (default: 1000)")
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


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    device = get_device(args.device)

    if len(args.clip_pretrained_list) != len(args.clip_models):
        print("ERROR: --clip_pretrained_list must have the same number "
              "of entries as --clip_models.", file=sys.stderr)
        sys.exit(1)

    epsilon = args.epsilon
    N = args.N
    use_temporal = args.accident_temporal is not None

    # ── Print config ─────────────────────────────────────────────────
    print("=" * 60)
    print("TTI-UAP: Temporal Trajectory Injection Attack")
    print("=" * 60)
    model_strs = [f"{m}({p})" for m, p in
                  zip(args.clip_models, args.clip_pretrained_list)]
    print(f"  CLIP models     : {', '.join(model_strs)}")
    print(f"  Target texts    : {args.target_texts}")
    if args.negative_texts:
        print(f"  Negative texts  : {args.negative_texts}")
        print(f"  lambda_neg      : {args.lambda_neg}")
    if use_temporal:
        print(f"  Temporal data   : {args.accident_temporal}")
        print(f"  lambda_traj     : {args.lambda_traj}")
        print(f"  lambda_trans    : {args.lambda_trans}")
    else:
        print("  Temporal losses : DISABLED (no --accident_temporal)")
    print(f"  Device          : {device}")
    print(f"  epsilon (L-inf) : {epsilon:.6f}  ({epsilon * 255:.1f}/255)")
    print(f"  alpha (step)    : {args.alpha:.6f}  ({args.alpha * 255:.1f}/255)")
    print(f"  N (UAP frames)  : {N}")
    if args.di_prob > 0:
        print(f"  DI-FGSM         : p={args.di_prob}, scale_low={args.di_scale_low}")
    print(f"  MI-FGSM mu      : {args.mu}")
    print(f"  Image size      : {args.image_size}")
    print(f"  Epochs          : {args.epochs}")
    print()

    # ── Load CLIP surrogates ─────────────────────────────────────────
    clip_extractors = []
    clip_text_targets = []
    clip_text_negatives = []

    for cname, cpretrained in zip(args.clip_models, args.clip_pretrained_list):
        print(f"Loading CLIP {cname} ({cpretrained}) ...")
        cfe = CLIPFeatureExtractor(cname, cpretrained, device)
        clip_extractors.append(cfe)

        tt = cfe.encode_target_texts(args.target_texts)
        clip_text_targets.append(tt)
        print(f"  text target dim={tt.shape[-1]}")

        if args.negative_texts:
            tn = cfe.encode_target_texts(args.negative_texts)
            clip_text_negatives.append(tn)
        else:
            clip_text_negatives.append(None)

    print(f"\n{len(clip_extractors)} CLIP model(s) loaded.\n")

    # ── Load temporal templates ──────────────────────────────────────
    traj_templates = {}
    trans_targets = {}

    if use_temporal:
        print(f"Loading temporal templates from {args.accident_temporal} ...")
        tdata = torch.load(args.accident_temporal, map_location=device,
                           weights_only=False)
        loaded_N = tdata["N"]
        if loaded_N != N:
            print(f"WARNING: temporal template N={loaded_N} != UAP N={N}. "
                  f"Templates will be interpolated.", file=sys.stderr)

        for cfe in clip_extractors:
            key = cfe.key
            if key in tdata["templates"]:
                tmpl = tdata["templates"][key].to(device)
                if tmpl.shape[0] != N:
                    tmpl = F.interpolate(
                        tmpl.unsqueeze(0).unsqueeze(0),
                        size=(N, tmpl.shape[1]),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0).squeeze(0)
                    tmpl = tmpl / tmpl.norm(dim=-1, keepdim=True)
                traj_templates[key] = tmpl
                print(f"  {key}: trajectory template {list(tmpl.shape)}")
            else:
                print(f"  {key}: NOT FOUND in temporal data, L_traj disabled")

            if "transition_sims" in tdata and key in tdata["transition_sims"]:
                tsim = tdata["transition_sims"][key].to(device)
                if tsim.shape[0] != N - 1:
                    tsim = F.interpolate(
                        tsim.unsqueeze(0).unsqueeze(0),
                        size=(1, N - 1),
                        mode="linear", align_corners=False,
                    ).squeeze(0).squeeze(0)
                trans_targets[key] = tsim
                print(f"  {key}: transition target {list(tsim.shape)}")
            else:
                print(f"  {key}: no transition sims, L_trans disabled")

        if tdata.get("avg_impact_position") is not None:
            print(f"  Impact position: {tdata['avg_impact_position']:.2%}")
        print()

    # ── Load source images ───────────────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(args.image_dir, transform=transform,
                          max_images=args.max_images)
    print(f"Source images: {len(dataset)}  ({args.image_dir})")
    if len(dataset) == 0:
        print("No images found. Exiting.")
        sys.exit(1)
    print()

    # ── Initialise UAP ───────────────────────────────────────────────
    delta_params = [
        torch.nn.Parameter(
            torch.zeros(3, args.image_size, args.image_size, device=device)
        )
        for _ in range(N)
    ]
    momentum = [torch.zeros_like(p.data) for p in delta_params]

    # ── DI-FGSM closure ─────────────────────────────────────────────
    di_fn = None
    if args.di_prob > 0:
        _img_sz = args.image_size
        _di_p = args.di_prob
        _di_lo = args.di_scale_low
        def di_fn(t):
            return input_diversity(t, _img_sz, _di_p, _di_lo)

    # ── Optimisation loop ────────────────────────────────────────────
    global_step = 0
    output_stem = Path(args.output).stem
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_alpha = args.alpha * (0.5 * (1.0 + math.cos(
            math.pi * (epoch - 1) / args.epochs)))

        desc = f"Epoch {epoch}/{args.epochs}"
        pbar = tqdm(dataset, desc=desc, unit="img")

        for x in pbar:
            x = x.unsqueeze(0).to(device)             # (1, 3, H, W)

            # Clear gradients for all frames
            for dp in delta_params:
                if dp.grad is not None:
                    dp.grad.zero_()

            # Build batched adversarial images: one per UAP frame
            delta_stack = torch.stack(delta_params)    # (N, 3, H, W)
            x_batch = x.expand(N, -1, -1, -1)         # (N, 3, H, W)
            x_adv = torch.clamp(x_batch + delta_stack, 0.0, 1.0)

            if di_fn is not None:
                x_adv = di_fn(x_adv)

            # ── Ensemble loss across CLIP models ──────────────
            loss = torch.tensor(0.0, device=device)

            for cfe, txt_tgt, txt_neg in zip(
                    clip_extractors, clip_text_targets, clip_text_negatives):
                feat = cfe(x_adv)                      # (N, D)

                # L_clip: per-frame targeted semantic shift
                l_clip = clip_targeted_loss(
                    feat, txt_tgt,
                    text_negative=txt_neg,
                    lambda_neg=args.lambda_neg,
                )

                model_loss = l_clip

                # L_traj: trajectory matching
                key = cfe.key
                if use_temporal and key in traj_templates:
                    l_traj = temporal_trajectory_loss(feat, traj_templates[key])
                    model_loss = model_loss + args.lambda_traj * l_traj

                # L_trans: transition pattern
                if use_temporal and key in trans_targets:
                    l_trans = temporal_transition_loss(feat, trans_targets[key])
                    model_loss = model_loss + args.lambda_trans * l_trans

                loss = loss + model_loss

            loss = loss / len(clip_extractors)
            loss.backward()

            # ── MI-FGSM update for ALL N frames ───────────────
            with torch.no_grad():
                for n_idx in range(N):
                    g = delta_params[n_idx].grad.data
                    g = g / (g.abs().mean() + 1e-8)
                    momentum[n_idx] = args.mu * momentum[n_idx] + g
                    delta_params[n_idx].data -= epoch_alpha * momentum[n_idx].sign()
                    delta_params[n_idx].data.clamp_(-epsilon, epsilon)

            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             alpha=f"{epoch_alpha * 255:.2f}/255")

            if args.save_every and global_step % args.save_every == 0:
                ckpt = torch.stack([p.data for p in delta_params]).cpu()
                ckpt_path = output_dir / f"{output_stem}_ckpt{global_step}.pt"
                torch.save(ckpt, ckpt_path)
                tqdm.write(f"  Checkpoint -> {ckpt_path}  "
                           f"(loss={loss.item():.4f})")

    # ── Save final UAP ───────────────────────────────────────────────
    final = torch.stack([p.data for p in delta_params]).cpu()
    torch.save(final, args.output)

    print(f"\n{'=' * 60}")
    print(f"UAP saved -> {args.output}")
    print(f"  Shape  : {list(final.shape)}  (N, C, H, W)")
    print(f"  L-inf  : {final.abs().max().item():.6f}  (budget: {epsilon:.6f})")
    print(f"{'=' * 60}")

    for cfe in clip_extractors:
        cfe.remove()


if __name__ == "__main__":
    main()
