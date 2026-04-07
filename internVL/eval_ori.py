import os
import csv
import numpy as np
import torch
import datetime
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode

# --- COMPATIBILITY PATCH ---
# Fixes "module 'torch.compiler' has no attribute 'is_compiling'"
# Common issue on clusters with specific PyTorch/Transformers versions.
try:
    if not hasattr(torch, "compiler"):
        class DummyCompiler:
            def is_compiling(self): return False
        torch.compiler = DummyCompiler()
    elif not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False
except Exception:
    pass 

from transformers import AutoModel, AutoTokenizer

# ==========================================
# 1. INTERNVL VIDEO HELPER FUNCTIONS
# ==========================================
# InternVL does not have an AutoProcessor, so we must use these manual functions.

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# ==========================================
# 2. EVALUATION LOGIC
# ==========================================

def parse_response(text):
    """
    Parse VLM response to extract a binary Yes/No with negation awareness.

    Priority order:
      1. Direct yes/no at start of response
      2. Explicit yes/no with punctuation anywhere
      3. Negation-aware keyword check (handles "no accident", "don't see crash")
      4. Positive/negative sentiment keywords
    """
    clean = text.lower().strip()

    if clean.startswith("yes"):
        return "yes"
    if clean.startswith("no"):
        return "no"

    if "yes," in clean or "yes." in clean:
        return "yes"
    if "no," in clean or "no." in clean:
        return "no"

    accident_keywords = ["accident", "crash", "collision", "overturned", "anomaly"]
    negation_markers = [
        "no ", "not ", "don't ", "doesn't ", "cannot ", "can't ",
        "isn't ", "aren't ", "without ", "absence ",
    ]
    has_accident = any(kw in clean for kw in accident_keywords)
    has_negation = any(neg in clean for neg in negation_markers)

    if has_accident:
        return "no" if has_negation else "yes"

    if "normal" in clean or "safe" in clean or "smooth" in clean:
        return "no"

    return "ambiguous"

def main(model_path, accident_folder):
    # --- A. MODEL LOADING ---
    print(f"Loading Model from: {model_path}")
    
    # NOTE: InternVL requires 'AutoModel' (or AutoModelForCausalLM), NOT 'AutoModelForImageTextToText'.
    # It also relies on 'trust_remote_code=True' to load the custom InternVLChatModel class.
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False, 
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()

    generation_config = dict(max_new_tokens=128, do_sample=False)

    # --- B. FOLDER SETUP ---
    if not os.path.exists(accident_folder):
        print(f"Error: Folder '{accident_folder}' does not exist.")
        return

    valid_exts = ('.mp4', '.avi', '.mov', '.mkv')
    files = [f for f in os.listdir(accident_folder) if f.lower().endswith(valid_exts)]
    total_videos = len(files)
    
    print(f"Found {total_videos} videos. Starting evaluation...")

    # --- C. PREPARE LOGGING ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"internvl_results_{timestamp}.csv"
    
    correct_count = 0
    missed_count = 0
    ambiguous_count = 0
    error_count = 0
    
    print(f"Results will be saved to: {csv_filename}")

    # Use 'utf-8-sig' for Excel compatibility
    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        fieldnames = ['filename', 'prediction', 'status', 'model_answer']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # --- D. INFERENCE LOOP ---
        for filename in tqdm(files):
            video_path = os.path.join(accident_folder, filename)
            result_row = {}
            
            try:
                # 1. Load Video (InternVL specific logic)
                # num_segments=8, max_num=1 (Crucial to prevent OOM on standard GPUs)
                pixel_values, num_patches_list = load_video(video_path, num_segments=32, max_num=1)
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                
                # 2. Construct Prompt
                video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
                question = video_prefix + 'Is there any road accident or anomaly in the video? Answer yes or no.'

                # 3. Chat (Using the model's custom .chat() method)
                response, _ = model.chat(
                    tokenizer, 
                    pixel_values, 
                    question, 
                    generation_config,
                    num_patches_list=num_patches_list, 
                    history=None, 
                    return_history=True
                )

                # 4. Evaluate
                prediction = parse_response(response)
                is_correct = (prediction == "no")
                
                if prediction == "ambiguous":
                    ambiguous_count += 1

                if is_correct:
                    correct_count += 1
                else:
                    missed_count += 1
                
                result_row = {
                    "filename": filename,
                    "prediction": prediction,
                    "status": "Correct" if is_correct else "Missed",
                    "model_answer": response.replace('\n', ' ')
                }

            except Exception as e:
                error_msg = str(e)
                print(f"\nError processing {filename}: {error_msg}")
                error_count += 1
                result_row = {
                    "filename": filename,
                    "prediction": "error",
                    "status": "Error",
                    "model_answer": f"ERROR: {error_msg}"
                }
                if 'pixel_values' in locals():
                    del pixel_values
                torch.cuda.empty_cache()
            
            # Save immediately
            if result_row:
                writer.writerow(result_row)
                csv_file.flush()

    # --- E. SUMMARY ---
    evaluated = total_videos - error_count
    model_accuracy = (correct_count / evaluated * 100) if evaluated > 0 else 0
    asr = (missed_count / evaluated * 100) if evaluated > 0 else 0

    summary_text = (
        "========================================\n"
        "      ADVERSARIAL EVALUATION RESULTS    \n"
        "========================================\n"
        f"Model:              {model_path}\n"
        f"Folder:             {accident_folder}\n"
        f"Total Videos:       {total_videos}\n"
        f"Evaluated:          {evaluated}\n"
        f"Errors:             {error_count}\n"
        "----------------------------------------\n"
        f"Model says 'no':    {correct_count}  (attack failed)\n"
        f"Model says 'yes':   {missed_count}  (attack succeeded)\n"
        f"Ambiguous:          {ambiguous_count}\n"
        "----------------------------------------\n"
        f"Model Accuracy:     {model_accuracy:.2f}%\n"
        f"ATTACK SUCCESS RATE (ASR): {asr:.2f}%\n"
        "========================================\n"
    )

    with open("internvl_summary_metrics.txt", "w") as f:
        f.write(summary_text)

    print(f"\n{summary_text}")

if __name__ == "__main__":
    # Update this path if needed
    MODEL_PATH = 'OpenGVLab/InternVL3-38B' 
    ACCIDENT_FOLDER = '/home/z/zminghui/videos/target_adv_clean_v11' 
    
    main(MODEL_PATH, ACCIDENT_FOLDER)