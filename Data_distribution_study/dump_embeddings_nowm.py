# dump_embeddings.py

import os
import numpy as np
import torch
from transformers import AutoModel

from dataset import get_train_val_loaders      
from synth_dataset import build_synth_dataloaders 
from utils import SEED


# ====== Config ======
PROCESSOR_NAME = "google/siglip-large-patch16-384"
MODEL_NAME_OR_PATH = "google/siglip-large-patch16-384"  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For debugging/visualization, you can limit how many samples you embed.
# Set to None to use all samples from the dataloader.
MAX_REAL_SAMPLES  = 100000
MAX_SYNTH_SAMPLES = 100000


def extract_image_embeddings(dataloader, model, max_samples=None):
    """
    Iterate over a DataLoader and extract image embeddings.

    Args:
        dataloader: DataLoader that yields batches with a "pixel_values" key.
        model:      Vision-language model with `get_image_features` method.
        max_samples: If not None, stop after collecting this many samples.

    Returns:
        A NumPy array of shape (N, D) containing image embeddings.
    """
    model.eval()
    all_embeds = []
    n_seen = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(DEVICE)  # (B, C, H, W)
            
            image_embeds = model.get_image_features(pixel_values=pixel_values)  # (B, D)

            # Normalize embeddings for stability in similarity / visualization
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

            all_embeds.append(image_embeds.cpu())
            n_seen += image_embeds.size(0)

            if max_samples is not None and n_seen >= max_samples:
                print(f"[info] Reached max_samples={max_samples}, stopping early.")
                break

    if not all_embeds:
        raise RuntimeError("No embeddings extracted. Check your dataloader and batch keys.")

    all_embeds = torch.cat(all_embeds, dim=0)  # (N, D)
    if max_samples is not None and all_embeds.size(0) > max_samples:
        all_embeds = all_embeds[:max_samples]

    print(f"[info] Final embedding shape: {all_embeds.shape}")
    return all_embeds.cpu().numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ====== 1. Load model ======
    print(f"[model] Loading model: {MODEL_NAME_OR_PATH}")
    model = AutoModel.from_pretrained(MODEL_NAME_OR_PATH)
    model.to(DEVICE)
    print("[model] Loaded and moved to device:", DEVICE)

    # ====== 2. Web images dataloader ======
    print("\n[web] Building web train/val loaders ...")
    web_train_loader, web_val_loader = get_train_val_loaders(
        batch_size=64, 
        tr_val_ratio=0.90,
        shards_glob="mtf2025_web_images_en/[0-9][0-9][0-9][0-9][0-9]_nowm.tar",
        num_workers=4,
        processor_name=PROCESSOR_NAME,
        shuffle_buffer=1000,
    )

    print("[web] Extracting embeddings from *val* split ...")
    web_embeds = extract_image_embeddings(
        dataloader=web_train_loader,
        model=model,
        max_samples=MAX_REAL_SAMPLES,
    )
    np.save("./visualization/web_nowm_embeds.npy", web_embeds)
    print("[web] Saved to ./visualization/web_nowm_embeds.npy")

    # ====== 3. Synthetic images dataloader ======
    print("\n[synth] Building synth train/val loaders ...")
    synth_train_loader, synth_val_loader = build_synth_dataloaders(
        batch_size=64,
        hf_dataset_name="jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth",
        train_val_ratio=0.95,
        processor_name=PROCESSOR_NAME,
        num_workers=4,
        shuffle_train=True,
        shuffle_val=False,
    )

    print("[synth] Extracting embeddings from *val* split ...")
    synth_embeds = extract_image_embeddings(
        dataloader=synth_train_loader,
        model=model,
        max_samples=MAX_SYNTH_SAMPLES,
    )
    np.save("./visualization/synth_embeds.npy", synth_embeds)
    print("[synth] Saved to ./visualization/synth_embeds.npy")

    print("\n[done] web_embeds.npy & synth_embeds.npy are ready.")


if __name__ == "__main__":
    main()