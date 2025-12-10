# translate_shards.py
import os
import glob
import json
import webdataset as wds
from tqdm import tqdm

from translator import translate_batch_to_english

IN_DIR = "mtf2025_web_images"       
OUT_DIR = "mtf2025_web_images_en"    
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 64


def extract_caption_and_json(sample):
    """
    extract caption and json dict from a webdataset sample
    """
    js = sample.get("json")

    # parse json
    if isinstance(js, dict):
        js_dict = js
    else:
        js_dict = {}
        if isinstance(js, bytes):
            try:
                js_dict = json.loads(js.decode("utf-8", errors="ignore"))
            except Exception:
                js_dict = {}
        elif isinstance(js, str):
            try:
                js_dict = json.loads(js)
            except Exception:
                js_dict = {}

    cap = js_dict.get("caption")
    if cap is None:
        cap_raw = sample.get("txt")
        if isinstance(cap_raw, bytes):
            cap = cap_raw.decode("utf-8", errors="ignore")
        elif isinstance(cap_raw, str):
            cap = cap_raw

    return cap, js_dict


def process_shard(in_tar: str, out_tar: str):
    sink = wds.TarWriter(out_tar)

    dataset = wds.WebDataset(in_tar, shardshuffle=False)

    buffer_samples = []
    buffer_caps = []
    buffer_jsons = []

    def flush_buffer():
        """Translate buffered captions and write out translated samples."""
        if not buffer_samples:
            return

        # Run batched translation
        caps_en = translate_batch_to_english(buffer_caps)

        print("[DEBUG] EXAMPLE:", buffer_caps[0], "###", caps_en[0])

        # Write all buffered samples to output tar
        for sample, js, cap_en in zip(buffer_samples, buffer_jsons, caps_en):
            if isinstance(cap_en, str) and cap_en.strip():
                js["caption_en"] = cap_en
                sample["json"] = json.dumps(js, ensure_ascii=False)
            sink.write(sample)

        buffer_samples.clear()
        buffer_caps.clear()
        buffer_jsons.clear()

    for sample in tqdm(dataset, desc=f"{os.path.basename(in_tar)}", unit="sample"):
        cap, js = extract_caption_and_json(sample)

        if isinstance(cap, str) and cap.strip():
            buffer_samples.append(sample)
            buffer_caps.append(cap)
            buffer_jsons.append(js)
        else:
            sink.write(sample)
            continue

        if len(buffer_samples) >= BATCH_SIZE:
            flush_buffer()

    flush_buffer()
    sink.close()


def main():
    shard_paths = sorted(glob.glob(os.path.join(IN_DIR, "00013.tar")))
    print(f"Found {len(shard_paths)} shards")

    for in_tar in tqdm(shard_paths, desc="Shards", unit="shard"):
        base = os.path.basename(in_tar)
        out_tar = os.path.join(OUT_DIR, base)
        print(f"\nPROCESS {in_tar} -> {out_tar}")
        process_shard(in_tar, out_tar)


if __name__ == "__main__":
    main()