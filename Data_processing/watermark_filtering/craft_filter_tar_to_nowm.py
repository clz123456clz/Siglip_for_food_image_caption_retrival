import os
import sys
import io
import tarfile
from typing import List

import torch
from PIL import Image
from craft_text_detector import Craft


VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")


def has_watermark_craft(image: Image.Image, craft_model: Craft, min_boxes: int = 1) -> bool:
    """
    Use CRAFT text detector to decide whether an image has watermark/text.
    """
    prediction = craft_model.detect_text(image)
    boxes = prediction.get("boxes", [])
    return len(boxes) >= min_boxes


def process_single_tar(tar_path: str, craft_model: Craft, min_boxes: int = 1):
    """
    Process ONE tar file:
    - Detect watermarks via CRAFT
    - Save only non-watermarked images into <input>_nowm.tar
    """
    if not os.path.exists(tar_path):
        print(f"[ERROR] Tar not found: {tar_path}")
        return

    base_dir = os.path.dirname(tar_path)
    base_name = os.path.basename(tar_path)
    name_no_ext, _ = os.path.splitext(base_name)
    out_tar_path = os.path.join(base_dir, name_no_ext + "_nowm.tar")

    print("=" * 70)
    print(f"[INFO] Processing TAR: {tar_path}")
    print(f"[INFO] Output TAR    : {out_tar_path}")

    kept = 0
    total = 0
    wm_count = 0

    with tarfile.open(tar_path, "r") as tar_in, \
         tarfile.open(out_tar_path, "w") as tar_out:

        for member in tar_in.getmembers():
            if not member.isfile():
                continue

            name_lower = member.name.lower()
            if not name_lower.endswith(VALID_EXTS):
                continue

            total += 1

            file_obj = tar_in.extractfile(member)
            if file_obj is None:
                continue

            img_bytes = file_obj.read()

            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to load image {member.name}: {e}")
                continue

            # -------- CRAFT detection --------
            try:
                wm = has_watermark_craft(img, craft_model, min_boxes=min_boxes)
            except Exception as e:
                print(f"[WARN] CRAFT crashed on {member.name}: {e}")
                continue

            if wm:
                wm_count += 1
                continue  # skip watermarked

            # -------- Save non-watermarked --------
            info = tarfile.TarInfo(name=member.name)
            info.size = len(img_bytes)
            tar_out.addfile(info, io.BytesIO(img_bytes))
            kept += 1

            if kept % 100 == 0:
                print(f"  processed={total}, kept(no_wm)={kept}, wm={wm_count}")

    print(f"[DONE] {tar_path}")
    print(f"  Total images         : {total}")
    print(f"  With watermark (text): {wm_count}")
    print(f"  Kept (no watermark)  : {kept}")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python craft_filter_tar_to_nowm.py 00000.tar 00001.tar 00002.tar")
        sys.exit(1)

    tar_list = sys.argv[1:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Load CRAFT model once
    print("[INFO] Loading CRAFT model...")
    craft = Craft(output_dir=None, crop_type="box", cuda=(device == "cuda"))
    print("[INFO] CRAFT model loaded.\n")

    # Process each tar
    for tar_path in tar_list:
        process_single_tar(tar_path, craft_model=craft, min_boxes=1)

    craft.unload_craftnet_model()
    craft.unload_refinenet_model()
    print("[INFO] Unloaded CRAFT models.")


if __name__ == "__main__":
    main()
