import os
import sys
import io
import tarfile
from enum import Enum

from typing import Optional, List, Tuple

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Disable FlashAttention warnings in some environments
os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"


class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"


def run_example(task_prompt: TaskType, image, text_input, model, processor, device):
    """
    Run Florence-2 inference for a given task prompt and image.
    Returns a parsed answer dictionary from processor.post_process_generation().
    """
    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
        use_cache=True,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt.value,
        image_size=(image.width, image.height),
    )
    return parsed_answer


def has_watermark_conservative(
    image: Image.Image,
    model,
    processor,
    device,
    min_valid_boxes: int = 1,
    min_area_ratio: float = 0.02,   # ignore very small boxes
    max_area_ratio: float = 0.5,    # ignore boxes covering almost whole image
    center_only: bool = True,       # only count boxes near image center
    on_error_as_watermark: bool = False,  # treat internal errors as watermark or not
) -> bool:
    """
    More conservative Florence-2 watermark detection.

    Returns True if we detect at least `min_valid_boxes` "reasonable" bounding
    boxes that likely correspond to watermarks (or large text).
    Otherwise returns False.

    "Reasonable" here means:
      - Area within [min_area_ratio, max_area_ratio] of the whole image area
      - (Optionally) box center inside the central 50% region of the image
    """
    text_input = "watermark"
    task_prompt = TaskType.OPEN_VOCAB_DETECTION

    try:
        parsed_answer = run_example(task_prompt, image, text_input, model, processor, device)
    except Exception as e:
        print(f"[WARN] Florence-2 crashed inside has_watermark_conservative: {e}")
        return on_error_as_watermark

    image_width, image_height = image.size
    total_image_area = image_width * image_height

    detection_key = "<OPEN_VOCABULARY_DETECTION>"

    if detection_key not in parsed_answer:
        # No detection results at all -> treat as no watermark
        return False

    det = parsed_answer[detection_key]
    if "bboxes" not in det or not det["bboxes"]:
        return False

    valid_boxes = 0

    for bbox in det["bboxes"]:
        x1, y1, x2, y2 = map(int, bbox)
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w == 0 or h == 0:
            continue

        area = w * h
        area_ratio = area / float(total_image_area)

        # 1) Filter out boxes that are too small or too large
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue

        # 2) Optionally, only keep boxes near the image center
        if center_only:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            # Define "center region" as 25%~75% in both width and height
            if not (image_width * 0.25 <= cx <= image_width * 0.75 and
                    image_height * 0.25 <= cy <= image_height * 0.75):
                continue

        valid_boxes += 1

    # Only consider it a watermark if there are enough "good" boxes
    return valid_boxes >= min_valid_boxes


def filter_tar_keep_no_watermark(
    tar_path: str,
    model,
    processor,
    device,
    out_suffix: str = "_nowm.tar",
    max_side: int = 1024,
    # conservative detection parameters:
    min_valid_boxes: int = 1,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.5,
    center_only: bool = True,
    on_error_as_watermark: bool = False,
):
    """
    Read an input tar (e.g., 00000.tar), keep ONLY images without detected watermark,
    and write them into a new tar (e.g., 00000_nowm.tar).

    Watermarks are detected using the conservative Florence-2 heuristic above.
    """
    if not os.path.exists(tar_path):
        print(f"[ERROR] Tar file not found: {tar_path}")
        return

    base_dir = os.path.dirname(tar_path)
    base_name = os.path.basename(tar_path)
    name_no_ext, _ = os.path.splitext(base_name)
    out_name = name_no_ext + out_suffix
    out_path = os.path.join(base_dir, out_name)

    print("=" * 80)
    print(f"[Tar] Input : {tar_path}")
    print(f"[Tar] Output: {out_path}")

    kept = 0
    total = 0
    wm_count = 0
    crash_count = 0

    # Open input tar (read) and output tar (write)
    with tarfile.open(tar_path, "r") as tar_in, tarfile.open(out_path, "w") as tar_out:
        for member in tar_in.getmembers():
            if not member.isfile():
                continue

            lower_name = member.name.lower()

            # Only process image files
            if not (lower_name.endswith(".jpg") or
                    lower_name.endswith(".jpeg") or
                    lower_name.endswith(".png") or
                    lower_name.endswith(".webp") or
                    lower_name.endswith(".bmp") or
                    lower_name.endswith(".tiff")):
                continue

            total += 1
            file_obj = tar_in.extractfile(member)
            if file_obj is None:
                continue

            # Read raw file bytes
            img_bytes = file_obj.read()

            # Load the image
            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"[WARN] Failed to open image {member.name}: {e}")
                continue

            # Optional: resize very large images to make Florence more robust
            try:
                if max_side is not None:
                    image.thumbnail((max_side, max_side), Image.LANCZOS)
            except Exception as e:
                print(f"[WARN] Failed to resize image {member.name}: {e}")

            # Florence-2 watermark check (conservative)
            try:
                wm = has_watermark_conservative(
                    image,
                    model,
                    processor,
                    device,
                    min_valid_boxes=min_valid_boxes,
                    min_area_ratio=min_area_ratio,
                    max_area_ratio=max_area_ratio,
                    center_only=center_only,
                    on_error_as_watermark=on_error_as_watermark,
                )
            except Exception as e:
                crash_count += 1
                print(f"[WARN] Florence-2 crashed on {member.name}: {e}")
                # If you prefer to keep these uncertain images, set wm = False.
                # For now we skip them to be safe.
                continue

            if wm:
                wm_count += 1
                continue  # skip images with watermark

            # Keep the image (write original bytes to output tar)
            kept += 1
            info = tarfile.TarInfo(name=member.name)
            info.size = len(img_bytes)
            tar_out.addfile(info, io.BytesIO(img_bytes))

            if kept % 100 == 0:
                print(
                    f"  processed={total}, kept(no_wm)={kept}, "
                    f"wm={wm_count}, crashed={crash_count}"
                )

    print(f"[Done] {tar_path}")
    print(f"  total images        : {total}")
    print(f"  with watermark      : {wm_count}")
    print(f"  kept (no watermark) : {kept}")
    print(f"  crashed (skipped)   : {crash_count}")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python florence_filter_tar_conservative.py /path/to/00000.tar [00001.tar 00002.tar ...]")
        sys.exit(1)

    tar_paths = sys.argv[1:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n[Step 1] Loading Florence-2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    print("Florence-2 loaded successfully.\n")

    # You can tweak these global parameters here if you want
    min_valid_boxes = 1       # increase to 2/3 to be even more conservative
    min_area_ratio = 0.02
    max_area_ratio = 0.5
    center_only = True
    on_error_as_watermark = False
    max_side = 1024

    # Process all tar files
    for tar_path in tar_paths:
        filter_tar_keep_no_watermark(
            tar_path,
            model,
            processor,
            device,
            out_suffix="_nowm.tar",
            max_side=max_side,
            min_valid_boxes=min_valid_boxes,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            center_only=center_only,
            on_error_as_watermark=on_error_as_watermark,
        )


if __name__ == "__main__":
    main()
