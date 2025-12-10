import os
import sys
import io
import tarfile
from enum import Enum

from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# 避免某些环境下的 FlashAttention 警告
os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"


# =========================
# Florence open-vocab 部分
# =========================

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"


def run_example(task_prompt: TaskType, image, text_input, model, processor, device):
    """
    调用 Florence-2 做一次 open-vocab 检测。
    这里对 post_process_generation 做了 try/except，
    防止出现 NoneType.shape 直接把整个程序搞崩。
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

    try:
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=task_prompt.value,
            image_size=(image.width, image.height),
        )
        return parsed_answer
    except Exception as e:
        # 这里就是你之前看到的 'NoneType' has no attribute 'shape' 那类错误
        print(f"[WARN] post_process_generation crashed: {e}")
        # 返回一个空 dict，让上游逻辑当成“没有检测到任何 bbox”
        return {}


def get_watermark_mask(image, model, processor, device):
    """
    复用你 inpainting 脚本里的逻辑：
    - 用 Florence-2 检测 "watermark" 对应的 bbox
    - 把 bbox 画到 mask 上（L 模式，255 = 可能有水印）
    - 同时返回一个 found_watermark 布尔值，只要有一个合理 bbox 就认为“检测到水印”。

    注意：这里虽然仍然加了中心矩形的 heuristic 到 mask 上，
    但 found_watermark 只由 Florence 的 bbox 决定。
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

    # Florence 检测出来的 bbox
    if detection_key in parsed_answer and "bboxes" in parsed_answer[detection_key]:
        bboxes = parsed_answer[detection_key]["bboxes"]
        print(f"[INFO] Found {len(bboxes)} potential watermark regions")
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            bbox_area = (x2 - x1) * (y2 - y1)

            # 过滤掉面积超过整图 80% 的大 bbox（大概率是误检）
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
                print(
                    f"  Region {idx+1}: skipped "
                    f"(area {bbox_area} > 80% of image {total_image_area})"
                )

    if not found_watermark:
        print("[INFO] No watermarks detected by Florence, mask from detector is empty.")

    # 中心区域 heuristic（只影响 mask，不影响 found_watermark）
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

    return mask, found_watermark


# =========================
# tar 过滤部分
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
    读取一个 tar（例如 00000.tar），对其中所有图片用 get_watermark_mask 检测：
      - 如果 Florence 检测到 watermark（found=True） → 认为有水印，丢弃
      - 如果 Florence 没检测到 → 认为无水印，保留

    输出新 tar：如 00000_nowm.tar。
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

            # 打开图片
            try:
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"[PROC] {member.name}: FAIL to open -> {e}")
                continue

            # 可选：缩放到最大边不超过 max_side，减轻 Florence 负担
            try:
                if max_side is not None and max(image.size) > max_side:
                    ratio = max_side / max(image.size)
                    new_size = (
                        int(image.size[0] * ratio),
                        int(image.size[1] * ratio),
                    )
                    image = image.resize(new_size, Image.LANCZOS)
                    print(f"[PROC] {member.name}: resized to {new_size}")
            except Exception as e:
                print(f"[PROC] {member.name}: FAIL to resize -> {e}")

            # 用 Florence 检测水印
            try:
                _, found = get_watermark_mask(image, model, processor, device)
                print(f"[PROC] {member.name}: DETECT_OK -> found_watermark={found}")
            except Exception as e:
                crash_count += 1
                print(f"[PROC] {member.name}: FLORENCE_CRASH -> {e}")
                # 如果 on_error_as_watermark=True，就把出错样本当“有水印”丢掉
                if on_error_as_watermark:
                    wm_count += 1
                    print(f"[PROC] {member.name}: SKIP (treat crash as watermark)")
                    continue
                else:
                    print(f"[PROC] {member.name}: KEEP (treat crash as no watermark)")
                    found = False

            if found:
                wm_count += 1
                print(f"[PROC] {member.name}: SKIP (watermark detected)")
                continue

            # 保存“认为没有水印”的图片到新 tar
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
        print("  python florence_filter_tar_with_mask.py 00000.tar [00001.tar ...]")
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
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True,
    )
    print("Florence-2 loaded successfully.\n")

    # 全局参数可以在这里调
    max_side = 1024
    on_error_as_watermark = True  # 出错时当“有水印”丢弃

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
