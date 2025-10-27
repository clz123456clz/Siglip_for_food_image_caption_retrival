import glob
import hashlib
from torch.utils.data import DataLoader
import webdataset as wds
from transformers import AutoProcessor



def extract_img_caption(sample):
    """
    Return a tuple (PIL.Image, str_caption).

    Given a WebDataset sample, prefer the caption from sample["json"]["caption"] if present
    and non-empty; otherwise fall back to sample["txt"]. The returned caption is guaranteed
    to be a clean `str` (never None).
    """

    img = sample.get("jpg") or sample.get("png")
    cap = None

    js = sample.get("json")
    if isinstance(js, dict):
        c = js.get("caption")
        if isinstance(c, str) and c.strip():
            cap = c

    if cap is None:
        cap_raw = sample.get("txt")
        if isinstance(cap_raw, bytes):
            cap = cap_raw.decode("utf-8", errors="ignore")
        elif isinstance(cap_raw, str):
            cap = cap_raw

    if cap is None:
        cap = "" 

    cap = cap.strip()
    return img, cap



def _make_split_filters(train_ratio: float):
    """
    Deterministically split samples into train/validation sets based on the MD5 hash of their __key__.

    Args:
        train_ratio (float): Proportion of samples to assign to the training set (0-1). 
            For example, 0.95 means 95% training and 5% validation.

    Returns:
        tuple[Callable, Callable]: 
            Two filter functions (filter_train, filter_val) that can be used with WebDataset `.select()`.
    """
    assert 0.0 < train_ratio < 1.0

    def _score_from_key(key: str) -> float:
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        v = int(h, 16) % 1000000  
        return v / 1000000.0

    def filter_train(sample: dict) -> bool:
        key = sample.get("__key__", "")
        return _score_from_key(key) < train_ratio

    def filter_val(sample: dict) -> bool:
        key = sample.get("__key__", "")
        return _score_from_key(key) >= train_ratio

    return filter_train, filter_val



# ========== main：return train / val two DataLoader ==========
def get_train_val_loaders(
    batch_size: int,
    tr_val_ratio: float,
    shards_glob: str = "mtf2025_web_images/*.tar",
    num_workers: int = 1,
    processor_name: str = "google/siglip-large-patch16-384",
    shuffle_buffer: int = 1000,
) -> tuple[DataLoader, DataLoader]:
    """
    Load multiple .tar shards (WebDataset), deterministically split them into train/validation sets,
    and return two DataLoaders.

    Args:
        batch_size (int): Number of samples per batch.
        tr_val_ratio (float): Proportion of data used for training (0-1). 
            For example, 0.95 means 95% training and 5% validation.
        shards_glob (str): Glob pattern pointing to the .tar shards to load.
        num_workers (int): Number of subprocesses for data loading.
        processor_name (str): The pretrained SigLIP processor name used for image/text preprocessing.
        shuffle_buffer (int): Buffer size for sample-level shuffling.

    Returns:
        tuple[DataLoader, DataLoader]: 
            A pair of (train_loader, val_loader).
    """
 
    shards = sorted(glob.glob(shards_glob))
    if len(shards) == 0:
        raise FileNotFoundError(f"No .tar shards found by glob: {shards_glob}")
    print(f"[dataset] found {len(shards)} shards, e.g. {shards[:3]} ...")

    processor = AutoProcessor.from_pretrained(processor_name)

    def collate(batch):
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


    f_train, f_val = _make_split_filters(tr_val_ratio)


    base_kwargs = dict(
        handler=wds.warn_and_continue,  
        shardshuffle=True             
    )

    # === Train Dataset ===
    train_ds = (
        wds.WebDataset(shards, **base_kwargs)
        .select(f_train)               
        .shuffle(shuffle_buffer)        
        .decode("pil")
        .map(extract_img_caption)
        .select(lambda x: x[0] is not None and isinstance(x[1], str) and len(x[1]) > 0)
    )

    # === Val Dataset ===
    val_ds = (
        wds.WebDataset(shards, **base_kwargs)
        .select(f_val)
        .shuffle(shuffle_buffer // 4) 
        .decode("pil")
        .map(extract_img_caption)
        .select(lambda x: x[0] is not None and isinstance(x[1], str) and len(x[1]) > 0)
    )

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=True,   
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=max(1, num_workers // 2),
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )

    print("[dataset] train/val loaders ready.")
    return train_loader, val_loader