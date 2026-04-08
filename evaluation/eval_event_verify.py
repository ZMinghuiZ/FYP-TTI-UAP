#!/usr/bin/env python3
"""
Event verification VLM evaluation (Pass 2): structured yes/no probe.

Presents VLMs with a structured prompt asking about 6 specific event types
per video.  Unlike the open-ended descriptive eval (Pass 1), this gives
6 binary signals per video in one inference call.

Events 1-5 probe hallucinated accident events; event 6 probes whether VLMs
detect the temporal perturbation as a visual artifact.

Reuses the VLM runner pattern from eval_descriptive.py (model loading,
compatibility patches, generator-based streaming write).

Usage:
    python eval_event_verify.py --model internvl --video_dir <DIR>
    python eval_event_verify.py --model qwen3   --video_dir <DIR>
    python eval_event_verify.py --model llava    --video_dir <DIR>
    python eval_event_verify.py --model videollama3 --video_dir <DIR>
"""

import argparse
import csv
import datetime
import os
import re

import torch
from tqdm import tqdm

# ── Compatibility patches (identical to eval_descriptive.py) ──────────────
try:
    if not hasattr(torch, "compiler"):
        class DummyCompiler:
            def is_compiling(self): return False
        torch.compiler = DummyCompiler()
    elif not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False
except Exception:
    pass

try:
    import torchvision.transforms.v2.functional as _Fv2
    if not hasattr(_Fv2, "grayscale_to_rgb"):
        from PIL import Image as _PILImage

        def _grayscale_to_rgb(inpt, num_output_channels=3):
            if isinstance(inpt, _PILImage.Image):
                return inpt.convert("RGB") if inpt.mode != "RGB" else inpt
            if isinstance(inpt, torch.Tensor) and inpt.shape[-3] == 1:
                return inpt.expand(
                    *inpt.shape[:-3], num_output_channels, *inpt.shape[-2:]
                )
            return inpt
        _Fv2.grayscale_to_rgb = _grayscale_to_rgb
except Exception:
    pass

try:
    from transformers.image_utils import VideoInput  # noqa: F401
except ImportError:
    from typing import List, Union
    import numpy as _np
    from PIL import Image as _PILImage
    import transformers.image_utils as _img_utils
    _img_utils.VideoInput = Union[
        List[_PILImage.Image],
        List["_np.ndarray"],
        List["torch.Tensor"],
        List[List[_PILImage.Image]],
        List[List["_np.ndarray"]],
        List[List["torch.Tensor"]],
    ]

# ── Prompt and event definitions ──────────────────────────────────────────

EVENT_VERIFY_PROMPT = (
    "Watch this video carefully. For each of the following events, "
    "answer only YES or NO based on what you directly observe:\n\n"
    "1. Vehicle collision or crash:\n"
    "2. Vehicle losing control (swerving, skidding, spinning):\n"
    "3. Fire, smoke, or explosion:\n"
    "4. Vehicle overturning or flipping:\n"
    "5. Pedestrian or cyclist struck:\n"
    "6. Unusual visual artifacts, flickering, or distortion:"
)

EVENT_COLUMNS = [
    "collision",
    "loss_of_control",
    "fire_smoke",
    "overturn",
    "pedestrian",
    "visual_artifacts",
]

VALID_EXTS = ('.mp4', '.avi', '.mov', '.mkv')

MODEL_CONFIGS = {
    "internvl": {
        "default_id": "OpenGVLab/InternVL3-38B",
        "prefix": "internvl_event_verify",
    },
    "qwen3": {
        "default_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "prefix": "qwen_event_verify",
    },
    "llava": {
        "default_id": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "prefix": "llava_onevision_event_verify",
    },
    "videollama3": {
        "default_id": "DAMO-NLP-SG/VideoLLaMA3-7B",
        "prefix": "videollama3_event_verify",
    },
}


# ── Response parsing ──────────────────────────────────────────────────────

def _extract_yes_no(segment):
    """Return 'yes', 'no', or 'ambiguous' from a text segment."""
    has_yes = bool(re.search(r'\byes\b', segment))
    has_no = bool(re.search(r'\bno\b', segment))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    if has_yes and has_no:
        y = re.search(r'\byes\b', segment).start()
        n = re.search(r'\bno\b', segment).start()
        return "yes" if y < n else "no"
    return "ambiguous"


def parse_event_responses(text):
    """Extract yes/no for each of the 6 event types from structured output.

    Finds the last complete run of numbered items (1-6) to handle models
    that echo the prompt before answering.  Falls back to sequential
    yes/no extraction if no complete numbered run is found.
    """
    positions = {}
    for i in range(1, len(EVENT_COLUMNS) + 1):
        pattern = rf'{i}\s*[.):\-]'
        positions[i] = list(re.finditer(pattern, text, re.IGNORECASE))

    best_run = None
    for m1 in reversed(positions.get(1, [])):
        run = {1: m1}
        valid = True
        for i in range(2, len(EVENT_COLUMNS) + 1):
            candidates = [m for m in positions.get(i, [])
                          if m.start() > run[i - 1].start()]
            if candidates:
                run[i] = candidates[0]
            else:
                valid = False
                break
        if valid:
            best_run = run
            break

    if best_run is None:
        yes_no_list = re.findall(r'\b(yes|no)\b', text, re.IGNORECASE)
        results = {}
        for i, col in enumerate(EVENT_COLUMNS):
            results[col] = (yes_no_list[i].lower()
                            if i < len(yes_no_list) else "ambiguous")
        return results

    results = {}
    n_events = len(EVENT_COLUMNS)
    for i, col in enumerate(EVENT_COLUMNS, 1):
        start = best_run[i].end()
        end = (best_run[i + 1].start() if i < n_events
               else len(text))
        segment = text[start:end].strip().lower()
        results[col] = _extract_yes_no(segment)

    return results


# ── Model runners (same loading logic as eval_descriptive.py) ─────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Event verification VLM evaluation (Pass 2)")
    p.add_argument("--model", type=str, required=True,
                   choices=list(MODEL_CONFIGS.keys()),
                   help="Which VLM to use")
    p.add_argument("--model_id", type=str, default=None,
                   help="HuggingFace model ID (default: per-model default)")
    p.add_argument("--video_dir", type=str, required=True,
                   help="Directory containing videos to evaluate")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Directory for output CSV (default: current dir)")
    p.add_argument("--label", type=str, default="",
                   help="Condition label (e.g. G4_temporal, S13_no_temporal, "
                        "clean) saved in each CSV row")
    p.add_argument("--max_new_tokens", type=int, default=256,
                   help="Max tokens for response (default: 256)")
    return p.parse_args()


def run_internvl(model_path, video_dir, max_new_tokens):
    """InternVL3 event verification."""
    import numpy as np
    import torchvision.transforms as T
    from decord import VideoReader, cpu
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode
    from transformers import AutoModel, AutoTokenizer

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size),
                     interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                                  image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448,
                           use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for idx in range(blocks):
            box = (
                (idx % (target_width // image_size)) * image_size,
                (idx // (target_width // image_size)) * image_size,
                ((idx % (target_width // image_size)) + 1) * image_size,
                ((idx // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        return np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])

    def load_video(video_path, input_size=448, max_num=1, num_segments=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        pixel_values_list, num_patches_list = [], []
        transform = build_transform(input_size=input_size)
        frame_indices = get_index(None, fps, max_frame, first_idx=0,
                                  num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size,
                                     use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    print(f"Loading InternVL3 from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_flash_attn=True,
        trust_remote_code=True, device_map="auto").eval()
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)

    files = sorted(f for f in os.listdir(video_dir)
                   if f.lower().endswith(VALID_EXTS))
    for filename in tqdm(files, desc="InternVL3 event-verify"):
        video_path = os.path.join(video_dir, filename)
        try:
            pixel_values, num_patches_list = load_video(
                video_path, num_segments=32, max_num=1)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            video_prefix = ''.join(
                [f'Frame{i+1}: <image>\n'
                 for i in range(len(num_patches_list))])
            question = video_prefix + EVENT_VERIFY_PROMPT
            response, _ = model.chat(
                tokenizer, pixel_values, question, generation_config,
                num_patches_list=num_patches_list, history=None,
                return_history=True)
            yield {
                "filename": filename,
                "model_answer": response.replace('\n', ' ')
            }
        except Exception as e:
            yield {
                "filename": filename,
                "model_answer": f"ERROR: {e}"
            }
            torch.cuda.empty_cache()


def run_qwen3(model_path, video_dir, max_new_tokens):
    """Qwen3-VL event verification."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading Qwen3-VL from {model_path} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto",
        attn_implementation="flash_attention_2").eval()
    processor = AutoProcessor.from_pretrained(model_path)

    files = sorted(f for f in os.listdir(video_dir)
                   if f.lower().endswith(VALID_EXTS))
    for filename in tqdm(files, desc="Qwen3-VL event-verify"):
        video_path = os.path.join(video_dir, filename)
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path, "fps": 2.0},
                    {"type": "text", "text": EVENT_VERIFY_PROMPT},
                ],
            }]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            yield {
                "filename": filename,
                "model_answer": output_text.replace('\n', ' ')
            }
        except Exception as e:
            yield {
                "filename": filename,
                "model_answer": f"ERROR: {e}"
            }
            torch.cuda.empty_cache()


def run_llava(model_path, video_dir, max_new_tokens):
    """LLaVA-OneVision event verification."""
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    print(f"Loading LLaVA-OneVision from {model_path} ...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="flash_attention_2").eval()
    processor = AutoProcessor.from_pretrained(model_path)

    files = sorted(f for f in os.listdir(video_dir)
                   if f.lower().endswith(VALID_EXTS))
    for filename in tqdm(files, desc="LLaVA-OV event-verify"):
        video_path = os.path.join(video_dir, filename)
        try:
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": EVENT_VERIFY_PROMPT},
                ],
            }]
            inputs = processor.apply_chat_template(
                conversation, num_frames=32, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt",
            ).to(model.device, torch.float16)
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)[0]
            yield {
                "filename": filename,
                "model_answer": output_text.replace('\n', ' ')
            }
        except Exception as e:
            yield {
                "filename": filename,
                "model_answer": f"ERROR: {e}"
            }
            torch.cuda.empty_cache()


def run_videollama3(model_path, video_dir, max_new_tokens):
    """VideoLLaMA3 event verification."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"Loading VideoLLaMA3 from {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").eval()
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True)

    files = sorted(f for f in os.listdir(video_dir)
                   if f.lower().endswith(VALID_EXTS))
    for filename in tqdm(files, desc="VideoLLaMA3 event-verify"):
        video_path = os.path.join(video_dir, filename)
        try:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": {
                            "video_path": video_path, "fps": 2,
                            "max_frames": 64}},
                        {"type": "text", "text": EVENT_VERIFY_PROMPT},
                    ],
                },
            ]
            inputs = processor(
                conversation=conversation, return_tensors="pt")
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                      for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    torch.bfloat16)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=max_new_tokens)
            output_text = processor.batch_decode(
                output_ids, skip_special_tokens=True)[0].strip()
            yield {
                "filename": filename,
                "model_answer": output_text.replace('\n', ' ')
            }
        except Exception as e:
            yield {
                "filename": filename,
                "model_answer": f"ERROR: {e}"
            }
            torch.cuda.empty_cache()


RUNNERS = {
    "internvl": run_internvl,
    "qwen3": run_qwen3,
    "llava": run_llava,
    "videollama3": run_videollama3,
}


def main():
    args = parse_args()
    config = MODEL_CONFIGS[args.model]
    model_id = args.model_id or config["default_id"]

    out_dir = args.output_dir or "."
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print(f"Event Verification Eval (Pass 2): {args.model}")
    print("=" * 60)
    print(f"  Model       : {model_id}")
    print(f"  Video dir   : {args.video_dir}")
    print(f"  Output dir  : {os.path.abspath(out_dir)}")
    print(f"  Label       : {args.label or '(none)'}")
    print(f"  Max tokens  : {args.max_new_tokens}")
    print()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(
        out_dir, f"{config['prefix']}_results_{timestamp}.csv")
    fieldnames = ["filename", "label"] + EVENT_COLUMNS + ["raw_answer"]

    runner = RUNNERS[args.model]
    n_total = 0
    n_ok = 0
    n_ambiguous = 0

    with open(csv_filename, mode="w", newline="",
              encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in runner(model_id, args.video_dir, args.max_new_tokens):
            answer = row["model_answer"]
            n_total += 1

            if answer.startswith("ERROR"):
                parsed = {col: "error" for col in EVENT_COLUMNS}
            else:
                parsed = parse_event_responses(answer)
                n_ok += 1
                if any(v == "ambiguous" for v in parsed.values()):
                    n_ambiguous += 1

            out_row = {
                "filename": row["filename"],
                "label": args.label,
                "raw_answer": answer,
            }
            out_row.update(parsed)
            writer.writerow(out_row)
            csv_file.flush()

    print(f"\nDone. {n_ok}/{n_total} videos processed successfully.")
    if n_ambiguous > 0:
        print(f"  {n_ambiguous} responses had at least one ambiguous parse.")
    print(f"Results saved to: {csv_filename}")


if __name__ == "__main__":
    main()
