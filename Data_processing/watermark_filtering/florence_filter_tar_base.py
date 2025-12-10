import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import os
import sys
import io
import tarfile
from enum import Enum

from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# Make sure Python can find the local dummy flash_attn module
# (flash_attn/__init__.py) in this project directory.
sys.path.append(os.path.dirname(__file__))

os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"


# =========================
# Florence open-vocab part
# =========================

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"


def run_example(task_prompt: TaskType, image, text_input, model, processor, device):
    """
    Run Florence-2 (base) open-vocabulary detection once.

    We wrap post_process_generation in try/except so that
    any internal error will not crash the whole script.
    """
    prompt = task_prompt.value if text_input is None else task_prompt.value + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=128,
        early_stopping=True,
        do_sample=False,
        num_beams=1,
        use_cache=True,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    try:
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt.value,
            image_size=(image.width, image.height),
        )
        return parsed_answer
    except Exception as e:
        # This catches things like "NoneType has no attribute 'shape'"
        print(f"[WARN] post_process_generation crashed: {e}")
        return {}


def detect_watermark(
    image: Image.Image,
    model,
    processor,
    device: str,
    area_ratio_max: float = 0.9,
):
    """
    Use Florence-2-base to detect watermark in the image.

    Returns:
        has_wm (bool): True if at least one reasonable "watermark" bbox detected.
        boxes (list): list of [x1, y1, x2, y2] for detected watermarks.
    """
    text_input = "watermark"
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    parsed_answer = run_example(task_prompt, image, text_input, model, processor, device)

    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    if detection_key not in parsed_answer:
        return False, []

    det = parsed_answer[detection_key]
    bboxes = det.get("bboxes", [])
    labels = det.get("bboxes_labels", [])

    if not bboxes or not labels:
        return False, []

    image_width, image_height = image.size
    total_area = float(image_width * image_height)

    kept_boxes = []
    for bbox, label in zip(bboxes, labels):
        # Only care about "watermark" label
        if str(label).lower() != "watermark":
            continue

        x1, y1, x2, y2 = map(float, bbox)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            continue

        area = w * h
        area_ratio = area / total_area
        # Filter out extremely large boxes (almost full image)
        if area_ratio > area_ratio_max:
            continue

        kept_boxes.append([x1, y1, x2, y2])

    has_wm = len(kept_boxes) > 0
    return has_wm, kept_boxes


# =========================
# tar filtering part
# =========================

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")


def filter_tar_keep_no_watermark(
    tar_path: str,
    model,
    processor,
    device: str,
    out_suffix: str = "_nowm.tar",
    max_side: int = 1024,
    on_error_as_watermark: bool = True,
):
    """
    Read an input tar (e.g., 00000.tar), detect watermark on each image using
    Florence-2-base, and keep ONLY images without detected watermark.

    Output:   00000_nowm.tar  (same folder as input tar)
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

    with tarfile.open(tar_path, "r") as tar_in, \
         tarfile.open(out_path, "w") as tar_out:

        for member in tar_in.getmembers():
            if not member.isfile():
                continue

            lower_name = member.name.lower()
            if not lower_name.endswith(VALID_EXTS):
                continue

            total += 1
            print(f"[PROC] {member.name}: loading...")

            file_obj = tar_in.extractfile(member)
            if file_obj is None:
                print(f"[PROC] {member.name}: FAIL to extract")
                continue

            img_bytes = file_obj.read()

            # Open image
            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"[PROC] {member.name}: FAIL to open -> {e}")
                continue

            # Optional: resize large images to reduce load
            try:
                if max_side is not None and max(image.size) > max_side:
                    ratio = max_side / float(max(image.size))
                    new_size = (
                        int(image.size[0] * ratio),
                        int(image.size[1] * ratio),
                    )
                    image = image.resize(new_size, Image.LANCZOS)
                    print(f"[PROC] {member.name}: resized to {new_size}")
            except Exception as e:
                print(f"[PROC] {member.name}: FAIL to resize -> {e}")

            # Run Florence detection
            try:
                has_wm, boxes = detect_watermark(image, model, processor, device)
                print(f"[PROC] {member.name}: DETECT_OK -> has_watermark={has_wm}, boxes={boxes}")
            except Exception as e:
                crash_count += 1
                print(f"[PROC] {member.name}: FLORENCE_CRASH -> {e}")
                if on_error_as_watermark:
                    wm_count += 1
                    print(f"[PROC] {member.name}: SKIP (treat crash as watermark)")
                    continue
                else:
                    print(f"[PROC] {member.name}: KEEP (treat crash as no watermark)")
                    has_wm = False

            if has_wm:
                wm_count += 1
                print(f"[PROC] {member.name}: SKIP (watermark detected)")
                continue

            # Save original bytes (no-watermark image) into new tar
            info = tarfile.TarInfo(name=member.name)
            info.size = len(img_bytes)
            tar_out.addfile(info, io.BytesIO(img_bytes))
            kept += 1
            print(f"[PROC] {member.name}: SAVED (no watermark)")

    print(f"[Done] {tar_path}")
    print(f"  total images   : {total}")
    print(f"  with watermark : {wm_count}")
    print(f"  kept (no wm)   : {kept}")
    print(f"  crashed        : {crash_count}")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python florence_filter_tar_base.py 00000.tar [00001.tar 00002.tar ...]")
        sys.exit(1)

    tar_paths = sys.argv[1:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n[Step 1] Loading Florence-2-base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
        attn_implementation="eager",  # do not force flash_attn
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base",
        trust_remote_code=True,
    )
    print("Florence-2-base loaded successfully.\n")

    # Global parameters
    max_side = 512
    on_error_as_watermark = True  # safer: treat internal errors as "watermarked" and drop

    for tar_path in tar_paths:
        filter_tar_keep_no_watermark(
            tar_path,
            model,
            processor,
            device,
            out_suffix="_nowm.tar",
            max_side=max_side,
            on_error_as_watermark=on_error_as_watermark,
        )


if __name__ == "__main__":
    main()
