import torch
from transformers import AutoProcessor, AutoModel, SiglipModel
from PIL import Image
import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_matrix, save_npz
import os
from utils import config
from train import restore_checkpoint
import json

# === CONFIG ===
image_dir = "../Test2/imgs"
image_list_file = "../Test2/images.txt"
caption_file = "../Test2/captions.json"
image_batch_size = 1024
caption_batch_size = 8192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = config("proj_head_en_v2_t2_qv")

 # === LOAD CAPTIONS ===
with open(caption_file, "r") as f:
    all_captions = json.load(f)
num_captions = len(all_captions)
print(f"✅ Loaded {num_captions} captions.")    

# === LOAD IMAGE PATHS ===
with open(image_list_file, "r") as f:
    image_filenames = [line.strip() for line in f if line.strip()]
image_paths = [os.path.join(image_dir, fname) for fname in image_filenames]
num_images = len(image_paths)
print(f"✅ Loaded {num_images} image paths.")

# === LOAD MODEL & PROCESSOR ===
model_name = cfg["model_name"]
processor = AutoProcessor.from_pretrained(model_name)
model = SiglipModel.from_pretrained(model_name)
model.to(device)
model.eval()

# === Restore Checkpoints from Best Epoch ===
checkpoint_dir = cfg["checkpoint_dir"]
model, start_epoch, _ = restore_checkpoint(model, checkpoint_dir, True, True, False)
print(f"✅ Model restored. Resume from epoch {start_epoch}.")

# === INFERENCE LOOP ===
row_indices = []
col_indices = []

for image_start in tqdm(range(0, num_images, image_batch_size), desc="Image Batches"):
    image_end = min(image_start + image_batch_size, num_images)
    batch_paths = image_paths[image_start:image_end]
    
    # Load images
    image_batch = []
    valid_indices = []
    for idx, path in enumerate(batch_paths):
        try:
            img = Image.open(path).convert("RGB")
            image_batch.append(img)
            valid_indices.append(idx)
        except Exception as e:
            print(f"⚠️ Failed to load image {path}: {e}")
            continue

    if not image_batch:
        continue

    all_logits = []

    for cap_start in range(0, num_captions, caption_batch_size):
        cap_end = min(cap_start + caption_batch_size, num_captions)
        caption_batch = all_captions[cap_start:cap_end]

        inputs = processor(
            text=caption_batch,
            images=image_batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image  # [num_valid_images, caption_batch_size]

        all_logits.append(logits.cpu())

    full_logits = torch.cat(all_logits, dim=1)  # [num_valid_images, num_captions]
    probs = torch.sigmoid(full_logits)

    # Single-label prediction: get index of most probable caption per image
    top1_indices = torch.argmax(probs, dim=1).numpy()

    # Collect row (image) and col (caption) indices
    for i, col_idx in enumerate(top1_indices):
        row_idx = image_start + valid_indices[i]
        row_indices.append(row_idx)
        col_indices.append(col_idx)

    # === DEBUG: sample inspection (for single-label case) ===
    if image_start == 0:  
        sample_ids = [0, 1, 2, 3, 4] 
        for sid in sample_ids:
            col = int(col_indices[sid])  # single caption index
            score = float(probs[sid, col].cpu())  # score of this caption
            print(f"\n🖼️ Image #{image_start + sid}:")
            print(f"  caption_idx={col:<4} | score={score:.4f} | text='{all_captions[col]}'")
# === BUILD SPARSE MATRIX ===
data = np.ones(len(row_indices), dtype=int)
sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_images, num_captions))


# === SAVE OUTPUT ===
save_npz("./results/sigliplarge384_single_lora_proj_head_en_v2_t2_r16_qv.npz", sparse_matrix)
print("✅ Saved single-label sparse matrix.")

