import os
import sys
import io
import tarfile
from enum import Enum

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


def has_watermark(image: Image.Image, model, processor, device, area_ratio_thresh=0.8) -> bool:
    """
    Run Florence-2 watermark detection and return True if a reasonable bounding box
    (not covering nearly the entire image) is detected.

    This version does NOT generate masks or apply heuristics; only detection is used.
    """
    text_input = "watermark"
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    parsed_answer = run_example(task_prompt, image, text_input, model, processor, device)

    image_width, image_height = image.size
    total_image_area = image_width * image_height

    detection_key = "<OPEN_VOCABULARY_DETECTION>"

    if detection_key not in parsed_answer:
        return False

    det = parsed_answer[detection_key]
    if "bboxes" not in det:
        return False

    # Check all detected bounding boxes
    for bbox in det["bboxes"]:
        x1, y1, x2, y2 = map(int, bbox)
        bbox_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Skip boxes too large (likely false positives)
        if bbox_area > 0 and bbox_area <= area_ratio_thresh * total_image_area:
            return True

    return False


def filter_tar_keep_no_watermark(
    tar_path: str,
    model,
    processor,
    device,
    out_suffix: str = "_nowm.tar",
):
    """
    Read an input tar (e.g., 00000.tar), keep ONLY images without detected watermark,
    and write them into a new tar (e.g., 00000_nowm.tar).
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
                    lower_name.endswith(".webp")):
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

            # Florence-2 watermark check
            try:
                wm = has_watermark(image, model, processor, device)
            except Exception as e:
                print(f"[WARN] Florence-2 crashed on {member.name}: {e}")
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
                print(f"  processed={total}, kept(no_wm)={kept}, wm={wm_count}")

    print(f"[Done] {tar_path}")
    print(f"  total images : {total}")
    print(f"  with watermark: {wm_count}")
    print(f"  kept (no wm) : {kept}")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python filter_no_watermark_from_tar.py /path/to/00000.tar [00001.tar 00002.tar ...]")
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

    # Process all tar files
    for tar_path in tar_paths:
        filter_tar_keep_no_watermark(tar_path, model, processor, device)


if __name__ == "__main__":
    main()
