import os
import numpy as np
from PIL import Image
import cv2  # pip install opencv-python

from dataset import get_train_val_datasets
from synth_dataset import MTF25SynthDataset

# ====== Config ======

# Real (web) dataset config: keep consistent with dump_embeddings.py
REAL_TR_VAL_RATIO = 0.95
REAL_SHARDS_GLOB = "mtf2025_web_images_en/[0-9][0-9][0-9][0-9][0-9].tar"
REAL_SHUFFLE_BUFFER = 10000

# Synthetic HF dataset name: keep consistent with your synth setup
SYNTH_DATASET_NAME = "jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth"

# How many images to analyze (you can reduce if it’s too slow)
MAX_REAL_IMAGES = 100000
MAX_SYNTH_IMAGES = 100000

# Output files
OUT_REAL_STATS = "./visualization/real_physical_stats.npz"
OUT_SYNTH_STATS = "./visualization/synth_physical_stats.npz"


# ====== Core stats computation (no geometry) ======

def compute_image_stats(image_iter, max_images=None):
    """
    Compute 'physical' statistics of images from an iterator of PIL.Image.

    brightness, color, and texture-related features.

    Args:
        image_iter: iterator yielding PIL.Image objects.
        max_images: if not None, stop after processing this many images.

    Returns:
        dict[str, np.ndarray]: each array has shape (N,), where N is the
        number of processed images.
    """

    brightness_means = []
    brightness_stds = []

    mean_R, mean_G, mean_B = [], [], []
    std_R, std_G, std_B = [], [], []

    mean_S, mean_V = [], []

    lap_var = []      # Laplacian variance (sharpness proxy)
    grad_mean = []    # mean gradient magnitude (edge strength proxy)

    count = 0

    for img in image_iter:
        # Stop if we have processed enough images
        count += 1
        if max_images is not None and count > max_images:
            break

        if img is None:
            continue

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Convert to numpy array
        arr = np.array(img)  # (H, W, 3), uint8

        # ----- brightness (grayscale) -----
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        brightness_means.append(gray.mean())
        brightness_stds.append(gray.std())

        # ----- RGB channel stats -----
        R = arr[:, :, 0]
        G = arr[:, :, 1]
        B = arr[:, :, 2]

        mean_R.append(R.mean())
        mean_G.append(G.mean())
        mean_B.append(B.mean())

        std_R.append(R.std())
        std_G.append(G.std())
        std_B.append(B.std())

        # ----- HSV stats (S, V) -----
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        Sc = hsv[:, :, 1]
        Vc = hsv[:, :, 2]

        mean_S.append(Sc.mean())
        mean_V.append(Vc.mean())

        # ----- Laplacian variance (sharpness) -----
        lap = cv2.Laplacian(gray, ddepth=cv2.CV_64F)
        lap_var.append(lap.var())

        # ----- Sobel gradient magnitude -----
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        grad_mean.append(grad_mag.mean())

        if count % 1000 == 0:
            print(f"[info] processed {count} images")

    print(f"[done] total images processed: {count}")

    stats = {
        "brightness_mean": np.array(brightness_means),
        "brightness_std": np.array(brightness_stds),

        "mean_R": np.array(mean_R),
        "mean_G": np.array(mean_G),
        "mean_B": np.array(mean_B),

        "std_R": np.array(std_R),
        "std_G": np.array(std_G),
        "std_B": np.array(std_B),

        "mean_S": np.array(mean_S),
        "mean_V": np.array(mean_V),

        "laplacian_var": np.array(lap_var),
        "grad_mean": np.array(grad_mean),
    }

    return stats


# ====== Real image iterator using your get_train_val_datasets ======

def iter_real_images():
    """
    Yield PIL.Image objects from your WebDataset pipeline.

    We reuse get_train_val_datasets from dataset.py, which already:
      - loads shards
      - decodes to PIL
      - maps to (image, caption) pairs
    Here we just take the image from (img, cap).
    """
    print("[real] Building real train/val datasets via get_train_val_datasets ...")
    train_ds, val_ds = get_train_val_datasets(
        tr_val_ratio=REAL_TR_VAL_RATIO,
        shards_glob=REAL_SHARDS_GLOB,
        shuffle_buffer=REAL_SHUFFLE_BUFFER,
    )
    print("[real] Datasets ready. Iterating over *train* split for stats ...")

    # You can choose train_ds instead if you prefer
    for img, cap in train_ds:
        # val_ds elements are (PIL.Image, caption_str)
        yield img


# ====== Synthetic image iterator using your MTF25SynthDataset ======

def iter_synth_images():
    """
    Yield PIL.Image objects from your synthetic HF dataset wrapper.

    We reuse MTF25SynthDataset from synth_dataset.py, which returns
    (PIL.Image, caption_str) in __getitem__.
    """
    print(f"[synth] Loading MTF25SynthDataset: {SYNTH_DATASET_NAME}")
    full_ds = MTF25SynthDataset(
        hf_split="train",
        hf_dataset_name=SYNTH_DATASET_NAME,
        download_mode=None,
    )
    print(f"[synth] Total examples in full HF dataset: {len(full_ds)}")

    for idx in range(len(full_ds)):
        img, cap = full_ds[idx]
        yield img


def main():
    # ----- Real images -----
    print("==== Computing physical stats for REAL images ====")
    real_iter = iter_real_images()
    real_stats = compute_image_stats(real_iter, max_images=MAX_REAL_IMAGES)

    os.makedirs(os.path.dirname(OUT_REAL_STATS), exist_ok=True)
    np.savez(OUT_REAL_STATS, **real_stats)
    print(f"[real] Stats saved to: {OUT_REAL_STATS}")

    # ----- Synthetic images -----
    print("\n==== Computing physical stats for SYNTHETIC images ====")
    synth_iter = iter_synth_images()
    synth_stats = compute_image_stats(synth_iter, max_images=MAX_SYNTH_IMAGES)

    os.makedirs(os.path.dirname(OUT_SYNTH_STATS), exist_ok=True)
    np.savez(OUT_SYNTH_STATS, **synth_stats)
    print(f"[synth] Stats saved to: {OUT_SYNTH_STATS}")

    print("\n[done] Physical stats for real & synthetic images have been computed.")


if __name__ == "__main__":
    main()