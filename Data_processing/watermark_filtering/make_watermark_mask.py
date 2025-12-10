import sys
import os
from enum import Enum

from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"


class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"


def run_example(task_prompt: TaskType, image, text_input, model, processor, device):
    """Runs an inference task using the model."""
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


def get_watermark_mask(image, model, processor, device):
    """
    Run watermark detection and return:
      - mask:   L-mode PIL image (255 on watermark regions, 0 elsewhere)
      - found:  bool, whether Florence detected any region

    Besides Florence's bboxes, we ALWAYS add a central rectangle heuristic
    (for very faint BIGSTOCK-like watermarks in the middle).
    """
    text_input = "watermark"
    task_prompt = TaskType.OPEN_VOCAB_DETECTION
    parsed_answer = run_example(task_prompt, image, text_input, model, processor, device)

    image_width, image_height = image.size
    total_image_area = image_width * image_height

    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)

    detection_key = "<OPEN_VOCABULARY_DETECTION>"
    found_watermark = False

    # 1) Florence-2 detected boxes
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        print(f"Found {len(parsed_answer[detection_key]['bboxes'])} potential watermark regions")
        for idx, bbox in enumerate(parsed_answer[detection_key]["bboxes"]):
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)

            # Skip bounding boxes that are too large (likely false positives)
            if bbox_area <= 0.8 * total_image_area:
                margin = 5
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(image_width, x2 + margin)
                y2 = min(image_height, y2 + margin)

                draw.rectangle([x1, y1, x2, y2], fill=255)
                found_watermark = True
                print(f"  Region {idx+1}: ({x1}, {y1}) -> ({x2}, {y2}) INCLUDED")
            else:
                print(f"  Region {idx+1}: skipped (area too large)")

    if not found_watermark:
        print("No watermarks detected or all regions were too large; mask from detector is empty.")

    # 2) Heuristic central band for faint BIGSTOCK-like watermark
    #    Make a rectangle centered in the image, with size proportional
    #    to the image size (e.g., 60% width, 35% height).
    center_width_ratio = 0.40
    center_height_ratio = 0.25

    cw = int(image_width * center_width_ratio)
    ch = int(image_height * center_height_ratio)

    cx = image_width // 2
    cy = image_height // 2

    cx1 = max(0, cx - cw // 2)
    cy1 = max(0, cy - ch // 2)
    cx2 = min(image_width, cx + cw // 2)
    cy2 = min(image_height, cy + ch // 2)

    draw.rectangle([cx1, cy1, cx2, cy2], fill=255)
    print(f"Added central heuristic box: ({cx1}, {cy1}) -> ({cx2}, {cy2})")

    return mask, found_watermark


def main():
    # Input:  ./sample_images/sample_*.jpg
    # Output: ./sample_images/mask_*.jpg
    if len(sys.argv) != 2:
        print("Usage: python make_watermark_mask_v2.py ./sample_images/sample_*.jpg(00000.jpg)")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"ERROR: Input image {input_path} does not exist.")
        sys.exit(1)

    # Build output path: mask_0000.jpg in the same folder
    folder, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    mask_path = os.path.join(folder, f"mask_{name.split('_')[-1]}{ext}")  # sample_00000 -> mask_00000
    replace_path = os.path.join(folder, f"to_replace_{name.split('_')[-1]}{ext}")  # sample_00000 -> to_replace_00000

    print("=" * 60)
    print("Watermark Mask Generator (Florence-2 + central band heuristic)")
    print("=" * 60)
    print(f"Input image : {input_path}")
    print(f"Output mask : {mask_path}")
    print(f"Output 'to replace' image : {replace_path}")

    image = Image.open(input_path).convert("RGB")
    print(f"Loaded image size: {image.size[0]} x {image.size[1]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Florence-2
    print("\n[Step 1/2] Loading Florence-2 model for watermark detection...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    print("Florence-2 loaded successfully!")

    # Detect watermarks and get mask
    print("\n[Step 2/2] Detecting watermarks and building mask...")
    mask, found = get_watermark_mask(image, model, processor, device)

    # Build RGB mask image: watermark pixels = original; others = black
    black_bg = Image.new("RGB", image.size, (0, 0, 0))
    replace_rgb = Image.composite(image, black_bg, mask)  # where mask=255, take image; else black
    white_bg = Image.new("RGB", image.size, (255, 255, 255))   # all white background
    black_bg = Image.new("RGB", image.size, (0, 0, 0))         # all black background
    masked_rgb = Image.composite(white_bg, black_bg, mask) 

    masked_rgb.save(mask_path, quality=95)
    print(f"\n✅ Mask image saved to: {mask_path}")
    replace_rgb.save(replace_path, quality=95)
    print(f"\n✅ 'To replace' image saved to: {replace_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()