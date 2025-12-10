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
image_dir = "../Test1/imgs"
image_list_file = "../Test1/images.txt"
caption_file = "../Test1/captions.txt"
image_batch_size = 1024
caption_batch_size = 8192
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = config("proj_head_en_v2_t2_qv")

# === LOAD CAPTIONS ===
with open(caption_file, "r") as f:
    all_captions = [line.strip() for line in f if line.strip()]
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
    for path in batch_paths:
        try:
            img = Image.open(path).convert("RGB")
            image_batch.append(img)
        except Exception as e:
            print(f"⚠️ Failed to load image {path}: {e}")
            continue

    all_logits = []

    for cap_start in range(0, num_captions, caption_batch_size):
        cap_end = min(cap_start + caption_batch_size, num_captions)
        caption_batch = all_captions[cap_start:cap_end]

        inputs = processor(
            text=caption_batch,
            images=image_batch,
            return_tensors="pt",
            padding="max_length"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image  # [num_images, caption_batch_size]

        all_logits.append(logits.cpu())

    full_logits = torch.cat(all_logits, dim=1)  # [num_images, num_captions]
    probs = torch.sigmoid(full_logits)

    # Top-k selection
    k = 5
    topk_indices = torch.topk(probs, k=k, dim=1).indices
    pred_matrix = torch.zeros_like(probs).scatter(1, topk_indices, 1).int().cpu().numpy()

    row_indices_batch, col_indices_batch = np.where(pred_matrix == 1)
    row_indices.extend(row_indices_batch + image_start)
    col_indices.extend(col_indices_batch)

    # === DEBUG: sample inspection ===
    if image_start == 0:  
        sample_ids = [0, 1, 2, 3, 4] 
        for sid in sample_ids:
            topk_cols = topk_indices[sid].cpu().tolist()
            topk_scores = probs[sid, topk_cols].cpu().tolist()
            print(f"\n🖼️ Image #{image_start + sid}:")
            for rank, (col, score) in enumerate(zip(topk_cols, topk_scores), start=1):
                print(f"  #{rank:<2} caption_idx={col:<4} | score={score:.4f} | text='{all_captions[col]}'")

# === BUILD SPARSE MATRIX ===
data = np.ones(len(row_indices), dtype=int)
sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_images, num_captions))

# === SAVE OUTPUT ===
save_npz("./results/sigliplarge384_multi_lora_proj_head_en_v2_t2_r16_qv.npz", sparse_matrix)
print("✅ Saved sparse matrix.")
