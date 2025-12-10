import os
import io
import tarfile
import json
from typing import List, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from datasets import load_dataset

import random
import torch
from transformers import AutoProcessor

from utils import SEED  

# Valid image extensions inside the web tar files
VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")


class MixedMTFDataset(Dataset):
    """
    Mixed dataset made of:
      1) Web images + captions from local web shards
         - Case A: original shards, e.g. 00000.tar, containing:
             * <key>.jpg
             * <key>.json  with field "caption_en"
         - Case B: watermark-free shards, e.g. 00000_nowm.tar, containing:
             * <key>.jpg only
           In this case, we recover caption_en from the corresponding
           original shard 00000.tar by reading its JSON sidecars.

      2) Synthetic images + captions from a Hugging Face dataset:
         'jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth'

    Each __getitem__ returns:
        (image, caption)           if return_source == False
        (image, caption, source)   if return_source == True

    where:
      - image: transformed image (if `transform` is provided), otherwise a PIL.Image
      - caption: non-empty string (for web: caption_en)
      - source: "web" or "synth"
    """

    def __init__(
        self,
        web_tar_dir: str,
        web_tar_names: List[str],
        synth_hf_name: str = "jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth",
        synth_split: str = "train",
        transform=None,
        return_source: bool = True,
    ):
        """
        Args:
            web_tar_dir:
                Directory where the web tar files are stored,
                e.g. "mtf2025_web_images_en".
                It can contain either:
                  - original shards: 00000.tar, 00001.tar, ...
                  - no-watermark shards: 00000_nowm.tar, 00001_nowm.tar, ...
                You can also pass a mixed list.

            web_tar_names:
                List of tar filenames you want to use for web data, e.g.
                  ["00000_nowm.tar", "00001_nowm.tar", ...].

            synth_hf_name:
                HF dataset name for synthetic images.

            synth_split:
                Split of HF dataset, default "train".

            transform:
                Optional transform applied to images (both web + synth).

            return_source:
                If True, __getitem__ returns an extra string
                indicating "web" or "synth".
        """
        super().__init__()

        self.web_tar_dir = web_tar_dir
        self.web_tar_names = web_tar_names
        self.transform = transform
        self.return_source = return_source

        # Web index: list of (tar_path_for_image, img_member_name, caption_str)
        # NOTE: we store the caption string directly in the index so we do NOT
        #       need to read JSON again during training.
        self.web_index: List[Tuple[str, str, str]] = []

        # Cache for opened tar files (per-process / per-worker)
        self._tar_cache = {}

        print("[MixedMTFDataset] Building web image index...")
        self._build_web_index()
        print(f"[MixedMTFDataset] Web image-caption pairs: {len(self.web_index)}")

        # Synthetic dataset from HF: prefer "caption_en", fallback "caption"
        print(f"[MixedMTFDataset] Loading synthetic dataset: {synth_hf_name} ({synth_split})")
        raw_synth = load_dataset(synth_hf_name, split=synth_split)
        self.synth_dataset = []
        for ex in raw_synth:
            if "caption_en" in ex and isinstance(ex["caption_en"], str) and ex["caption_en"].strip():
                cap = ex["caption_en"]
            else:
                cap = ex.get("caption", "")
            if isinstance(cap, str) and cap.strip():
                self.synth_dataset.append({
                    "image": ex["image"],
                    "caption": cap.strip(),
                })
        print(f"[MixedMTFDataset] Synthetic dataset size (with caption): {len(self.synth_dataset)}")
        print(f"[MixedMTFDataset] Total mixed samples: {len(self)}")

    # ------------------------------------------------------------------
    # Build web index (handles both *.tar and *_nowm.tar)
    # ------------------------------------------------------------------
    def _build_web_index(self):
        """
        For each shard name in self.web_tar_names:

        - If it ends with '_nowm.tar' (Case B):
            * Find the corresponding original shard '<id>.tar'
            * Read all JSON files from the original shard once, building
              a mapping:  '<key>.jpg' -> caption_en
            * Scan the *_nowm.tar shard and, for each image file, look up
              its base name in that mapping; if found, add to web_index.

        - Otherwise (Case A, original shard with JSON):
            * For each image file in '<id>.tar', find matching JSON sidecar,
              read caption_en, and add to web_index.
        """
        for tar_name in self.web_tar_names:
            tar_path = os.path.join(self.web_tar_dir, tar_name)
            if not os.path.exists(tar_path):
                print(f"  [WARN] Tar file not found: {tar_path}, skipping.")
                continue

            if tar_name.endswith("_nowm.tar"):
                # ---------------- Case B: *_nowm.tar ----------------
                shard_id = tar_name.split("_")[0]        # "00000_nowm.tar" -> "00000"
                orig_name = f"{shard_id}.tar"
                orig_path = os.path.join(self.web_tar_dir, orig_name)
                if not os.path.exists(orig_path):
                    print(f"  [WARN] Original shard not found for {tar_name}: {orig_path}, skip this shard.")
                    continue

                # Build a map from base filename to caption_en
                filename2cap = self._build_caption_map_from_original(orig_path)

                # Now scan *_nowm.tar, and only keep images that appear in the map
                try:
                    with tarfile.open(tar_path, "r") as nowm_tar:
                        for member in nowm_tar.getmembers():
                            if not member.isfile():
                                continue
                            lower_name = member.name.lower()
                            if not lower_name.endswith(VALID_EXTS):
                                continue

                            base_name = os.path.basename(member.name)
                            if base_name not in filename2cap:
                                continue
                            cap_en = filename2cap[base_name]
                            self.web_index.append((tar_path, member.name, cap_en))
                except Exception as e:
                    print(f"  [WARN] Failed to open nowm tar {tar_path}: {e}")
                    continue

            else:
                # ---------------- Case A: original *.tar ----------------
                try:
                    with tarfile.open(tar_path, "r") as tar:
                        members = {m.name: m for m in tar.getmembers() if m.isfile()}

                        for name, m in members.items():
                            lower_name = name.lower()
                            if not lower_name.endswith(VALID_EXTS):
                                continue

                            stem, _ = os.path.splitext(name)
                            json_name = stem + ".json"
                            jmember = members.get(json_name)
                            if jmember is None:
                                continue

                            try:
                                js_bytes = tar.extractfile(jmember).read()
                                js = json.loads(js_bytes)
                            except Exception:
                                continue

                            cap_en = js.get("caption_en", "")
                            if not (isinstance(cap_en, str) and cap_en.strip()):
                                continue

                            self.web_index.append((tar_path, name, cap_en.strip()))

                except Exception as e:
                    print(f"  [WARN] Failed to open tar {tar_path}: {e}")
                    continue

        print(f"[MixedMTFDataset] Web index built with {len(self.web_index)} image-caption pairs.")

    def _build_caption_map_from_original(self, orig_tar_path: str):
        """
        Build a mapping from image filename to caption_en for an original shard.

        We assume the original shard contains:
            <key>.jpg
            <key>.json  (with field 'caption_en')

        Returns:
            dict: { '<key>.jpg': 'caption_en string', ... }
        """
        filename2cap = {}
        try:
            with tarfile.open(orig_tar_path, "r") as tar:
                members = {m.name: m for m in tar.getmembers() if m.isfile()}

                for name, m in members.items():
                    # We only care about JSON files here
                    if not name.lower().endswith(".json"):
                        continue

                    stem, _ = os.path.splitext(name)  # '123456'
                    # image is usually '<stem>.jpg'
                    img_candidates = [stem + ext for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]]

                    try:
                        js_bytes = tar.extractfile(m).read()
                        js = json.loads(js_bytes)
                    except Exception:
                        continue

                    cap_en = js.get("caption_en", "")
                    if not (isinstance(cap_en, str) and cap_en.strip()):
                        continue

                    for img_name in img_candidates:
                        # we store only the basename, because *_nowm.tar
                        # usually does not contain folder prefixes
                        base = os.path.basename(img_name)
                        filename2cap[base] = cap_en.strip()

        except Exception as e:
            print(f"  [WARN] Failed to read original shard {orig_tar_path}: {e}")

        return filename2cap

    def __len__(self):
        # Total size = web pairs + synthetic pairs
        return len(self.web_index) + len(self.synth_dataset)

    # ------------------------------------------------------------------
    # Loading single samples (fast path at training time)
    # ------------------------------------------------------------------
    def _get_tar(self, tar_path: str) -> tarfile.TarFile:
        """
        Lazy-open and cache tar files per worker.

        Each DataLoader worker will have its own copy of the dataset object,
        so this cache is per-process and does not need locking.
        """
        tar = self._tar_cache.get(tar_path)
        if tar is None:
            tar = tarfile.open(tar_path, "r")
            self._tar_cache[tar_path] = tar
        return tar

    def _load_web_sample(self, idx: int) -> Tuple[Image.Image, str]:
        """
        Load a web image + caption by web_index.

        We only read the image bytes here; caption is already stored
        in the index (no JSON parsing during training).
        """
        tar_path, img_name, caption = self.web_index[idx]
        tar = self._get_tar(tar_path)

        img_member = tar.getmember(img_name)
        img_bytes = tar.extractfile(img_member).read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        return image, caption

    def _load_synth_sample(self, idx: int) -> Tuple[Image.Image, str]:
        """
        Load a synthetic image + caption from pre-filtered synth_dataset.
        """
        example = self.synth_dataset[idx]
        image = example["image"]          # PIL.Image
        caption = example["caption"]      # clean string
        return image, caption

    def __getitem__(self, idx: int):
        """
        Fetch a single sample from either web or synthetic part.

        We index web first [0 .. len(web_index)-1], then synthetic
        [len(web_index) .. len(web_index)+len(synth_dataset)-1].
        """
        if idx < len(self.web_index):
            image, caption = self._load_web_sample(idx)
            source = "web"
        else:
            synth_idx = idx - len(self.web_index)
            image, caption = self._load_synth_sample(synth_idx)
            source = "synth"

        if self.transform is not None:
            image = self.transform(image)

        if self.return_source:
            return image, caption, source
        else:
            return image, caption

    def __del__(self):
        """
        Best-effort clean up for opened tar files.
        """
        for tar in self._tar_cache.values():
            try:
                tar.close()
            except Exception:
                pass
        self._tar_cache.clear()


# ----------------------------------------------------------------------
# Helper for worker seeding
# ----------------------------------------------------------------------
def seed_worker(worker_id):
    """
    Make each DataLoader worker use a different, yet reproducible seed,
    to avoid duplicated shuffling across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------------------------
# Build train/val DataLoaders with total_pairs + ratios
# ----------------------------------------------------------------------
def get_mixed_train_val_loaders(
    total_pairs: int,
    tr_val_ratio: float,
    web_synth_ratio: float,
    batch_size: int,
    web_tar_dir: str,
    web_tar_names: List[str],
    synth_hf_name: str = "jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth",
    synth_split: str = "train",
    num_workers: int = 4,
    processor_name: str = "google/siglip-large-patch16-384",
):
    """
    Build train/val DataLoaders from the mixed dataset.

    Args:
        total_pairs: total number of (image, caption) pairs to use (train + val).
        tr_val_ratio: fraction of pairs used for training, e.g. 0.95 => 95% train.
        web_synth_ratio: fraction of pairs that should come from web data,
                         e.g. 0.7 => 70% web / 30% synth (approximately).
        batch_size: batch size for both train and val.
        web_tar_dir, web_tar_names, synth_hf_name, synth_split:
            arguments forwarded to MixedMTFDataset.
        num_workers: number of DataLoader workers.
        processor_name: SigLIP processor name for collate.

    Returns:
        (train_loader, val_loader)
    """
    assert 0.0 < tr_val_ratio < 1.0
    assert 0.0 <= web_synth_ratio <= 1.0

    dataset = MixedMTFDataset(
        web_tar_dir=web_tar_dir,
        web_tar_names=web_tar_names,
        synth_hf_name=synth_hf_name,
        synth_split=synth_split,
        transform=None,
        return_source=False,  # training only needs (img, caption)
    )

    n_web = len(dataset.web_index)
    n_synth = len(dataset.synth_dataset)

    desired_web = int(total_pairs * web_synth_ratio)
    desired_synth = total_pairs - desired_web

    actual_web = min(desired_web, n_web)
    actual_synth = min(desired_synth, n_synth)
    actual_total = actual_web + actual_synth

    if actual_total < total_pairs:
        print(
            f"[mixed loader] WARNING: requested total_pairs={total_pairs}, "
            f"but only {actual_total} available (web={actual_web}, synth={actual_synth})."
        )

    print(
        f"[mixed loader] Using {actual_web} web pairs and {actual_synth} synth pairs "
        f"(total={actual_total})."
    )

    rng = np.random.default_rng(SEED)

    web_indices_local = rng.choice(n_web, size=actual_web, replace=False).tolist()
    synth_indices_local = rng.choice(n_synth, size=actual_synth, replace=False).tolist()
    synth_indices_global = [idx + n_web for idx in synth_indices_local]

    all_indices = web_indices_local + synth_indices_global
    rng.shuffle(all_indices)

    n_train = int(len(all_indices) * tr_val_ratio)
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]

    print(
        f"[mixed loader] Split into train={len(train_indices)} and val={len(val_indices)} samples."
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    processor = AutoProcessor.from_pretrained(processor_name, use_fast=True)

    def collate_fn(batch):
        """
        Collate a list of (image, caption) pairs into a single batch,
        letting the SigLIP processor handle image + text processing.
        """
        imgs, caps = zip(*batch)

        safe_caps = []
        for c in caps:
            if isinstance(c, str):
                safe_caps.append(c)
            elif isinstance(c, bytes):
                safe_caps.append(c.decode("utf-8", errors="ignore"))
            else:
                safe_caps.append(str(c))

        enc = processor(
            images=list(imgs),
            text=safe_caps,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return enc

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    print("[mixed loader] train/val DataLoaders ready.")
    return train_loader, val_loader


# if __name__ == "__main__":
#     """
#     Sanity checks for the mixed dataset + loaders.

#     We test:
#       1) How many pairs in total, how many web vs synth.
#       2) Whether image & caption both exist (non-empty) for sampled items.
#       3) Whether the train/val DataLoaders work and return reasonable batches.
#     """
#     from torchvision import transforms as T

#     # -----------------------------
#     # 0. Basic config
#     # -----------------------------
#     web_dir = "mtf2025_web_images_en"
#     web_tars = [
#         "00000_nowm.tar",
#         "00001_nowm.tar",
#         "00002_nowm.tar",
#         "00003_nowm.tar",
#         "00004_nowm.tar",
#         "00005_nowm.tar",
#         "00006_nowm.tar",
#         "00007_nowm.tar",   
#         "00008_nowm.tar",
#         "00009_nowm.tar",
#         "00010_nowm.tar",
#         "00011_nowm.tar",
#         "00012_nowm.tar",
#         "00013_nowm.tar",
#     ]

#     # Small transform just for checking DataLoader
#     img_transform = T.Compose([
#         T.Resize((224, 224)),
#         T.ToTensor(),
#     ])

#     # -----------------------------
#     # 1. Build raw MixedMTFDataset
#     #    (no transform, return_source=True)
#     # -----------------------------
#     print("\n===== Building raw MixedMTFDataset for inspection =====")
#     raw_dataset = MixedMTFDataset(
#         web_tar_dir=web_dir,
#         web_tar_names=web_tars,
#         synth_hf_name="jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth",
#         synth_split="train",
#         transform=None,
#         return_source=True,   # we want to see "web" / "synth"
#     )

#     num_web   = len(raw_dataset.web_index)
#     num_synth = len(raw_dataset.synth_dataset)
#     num_total = len(raw_dataset)

#     print("\n[Count check]")
#     print(f"  #web   pairs : {num_web}")
#     print(f"  #synth pairs : {num_synth}")
#     print(f"  #total pairs : {num_total}")

#     # -----------------------------
#     # 2. Sample items to check that
#     #    image & caption both exist
#     # -----------------------------
#     print("\n===== Checking existence of image + caption =====")
#     import itertools
#     from PIL import Image

#     n_to_check = min(200, len(raw_dataset))  # don't scan too many for speed
#     missing_img = 0
#     missing_cap = 0

#     for i in range(n_to_check):
#         img, cap, src = raw_dataset[i]

#         # image must be a PIL image
#         if not isinstance(img, Image.Image):
#             missing_img += 1

#         # caption must be a non-empty string
#         if not (isinstance(cap, str) and cap.strip()):
#             missing_cap += 1

#         # print a few examples only
#         if i < 5:
#             print(f"  sample[{i}] source={src}, caption[:60]={cap[:60]!r}")

#     print("\n[Image/caption sanity]")
#     print(f"  checked {n_to_check} samples")
#     print(f"  missing_img count : {missing_img}")
#     print(f"  missing_cap count : {missing_cap}")

#     # Hard assertion – you can comment this out if you just want to see numbers
#     if missing_img == 0 and missing_cap == 0:
#         print("  -> All checked samples have image + non-empty caption ✅")
#     else:
#         print("  -> WARNING: some samples are missing image or caption ❌")

#     # -----------------------------
#     # 3. Build train/val DataLoaders
#     #    and fetch a couple of batches
#     # -----------------------------
#     print("\n===== Building train/val DataLoaders =====")

#     total_pairs   = 1000          # just for test; adjust as you like
#     tr_val_ratio  = 0.95
#     web_synth_rat = 1.0             # 70% web, 30% synth (approx)

#     train_loader, val_loader = get_mixed_train_val_loaders(
#         total_pairs=total_pairs,
#         tr_val_ratio=tr_val_ratio,
#         web_synth_ratio=web_synth_rat,
#         batch_size=8,
#         web_tar_dir=web_dir,
#         web_tar_names=web_tars,
#         synth_hf_name="jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth",
#         synth_split="train",
#         num_workers=2,
#         processor_name="google/siglip-large-patch16-384",
#     )

#     # Just to be safe, rebuild a dataset WITH transform and check DataLoader manually
#     print("\n===== Manual DataLoader check with transform =====")
#     transformed_dataset = MixedMTFDataset(
#         web_tar_dir=web_dir,
#         web_tar_names=web_tars,
#         synth_hf_name="jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth",
#         synth_split="train",
#         transform=img_transform,
#         return_source=True,
#     )

#     from torch.utils.data import DataLoader as TorchDataLoader

#     demo_loader = TorchDataLoader(
#         transformed_dataset,
#         batch_size=4,
#         shuffle=True,
#         num_workers=2,
#     )

#     for bi, batch in enumerate(demo_loader):
#         imgs, caps, srcs = batch
#         print(f"\n[Demo DataLoader] batch {bi}")
#         print("  images shape:", imgs.shape)
#         print("  captions[0]:", caps[0])
#         print("  sources      :", srcs)
#         if bi >= 1:
#             break

#     print("\nAll tests finished.\n")