import os
import sys
import io
import tarfile
from enum import Enum

from typing import Optional, List, Tuple

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"


class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"


def run_example(task_prompt: TaskType, image, text_input, model, processor, device):
    """Run Florence-2 inference for a given task prompt and image."""
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
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.5,
    center_only: bool = True,
    on_error_as_watermark: bool = False,
) -> bool:

    text_input = "watermark"
    task_prompt = TaskType.OPEN_VOCAB_DETECTION

    parsed_answer = run_example(task_prompt, image, text_input, model, processor, device)

    image_width, image_height = image.size
    total_image_area = image_width * image_height

    detection_key = "<OPEN_VOCABULARY_DETECTION>"

    if detection_key not in parsed_answer:
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

        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue

        if center_only:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if not (image_width * 0.25 <= cx <= image_width * 0.75 and
                    image_height * 0.25 <= cy <= image_height * 0.75):
                continue

        valid_boxes += 1

    return valid_boxes >= min_valid_boxes


def filter_tar_keep_no_watermark(
    tar_path: str,
    model,
    processor,
    device,
    out_suffix: str = "_nowm.tar",
    max_side: int = 1024,
    min_valid_boxes: int = 1,
    min_area_ratio: float = 0.02,
    max_area_ratio: float = 0.5,
    center_only: bool = True,
    on_error_as_watermark: bool = False,
):

    if not os.path.exists(tar_path):
        print(f"[ERROR] Tar not found: {tar_path}")
        return

    base_dir = os.path.dirname(tar_path)
    base_name = os.path.basename(tar_path)
    name_no_ext, _ = os.path.splitext(base_name)
    out_path = os.path.join(base_dir, name_no_ext + out_suffix)

    print("=" * 80)
    print(f"[Tar] Input : {tar_path}")
    print(f"[Tar] Output: {out_path}")

    kept = 0
    total = 0
    wm_count = 0
    crash_count = 0

    with tarfile.open(tar_path, "r") as tar_in, \
         tarfile.open(out_path, "w") as tar_out:

        for member in tar_in.getmembers():
            if not member.isfile():
                continue

            lower = member.name.lower()
            if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or
                    lower.endswith(".png") or lower.endswith(".webp") or
                    lower.endswith(".bmp") or lower.endswith(".tiff")):
                continue

            total += 1
            print(f"[PROC] {member.name}: loading...")

            file_obj = tar_in.extractfile(member)
            if file_obj is None:
                print(f"[PROC] {member.name}: FAIL extract")
                continue

            img_bytes = file_obj.read()

            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"[PROC] {member.name}: FAIL open -> {e}")
                continue

            try:
                if max_side:
                    image.thumbnail((max_side, max_side), Image.LANCZOS)
            except Exception as e:
                print(f"[PROC] {member.name}: FAIL resize -> {e}")

            # Florence-2
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
                print(f"[PROC] {member.name}: DETECT_OK -> watermarked={wm}")
            except Exception as e:
                crash_count += 1
                print(f"[PROC] {member.name}: FLORENCE_CRASH -> {e}")
                continue

            if wm:
                wm_count += 1
                print(f"[PROC] {member.name}: SKIP (watermark)")
                continue

            # Save image
            info = tarfile.TarInfo(name=member.name)
            info.size = len(img_bytes)
            tar_out.addfile(info, io.BytesIO(img_bytes))
            kept += 1
            print(f"[PROC] {member.name}: SAVED (no watermark)")

    print(f"[Done] {tar_path}")
    print(f"  total      : {total}")
    print(f"  watermark  : {wm_count}")
    print(f"  kept       : {kept}")
    print(f"  crashed    : {crash_count}")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python florence_filter_tar_conservative_verbose.py 00000.tar 00001.tar ...")
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

    # Tuning parameters
    min_valid_boxes = 1
    min_area_ratio = 0.02
    max_area_ratio = 0.5
    center_only = True
    on_error_as_watermark = False
    max_side = 1024

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
