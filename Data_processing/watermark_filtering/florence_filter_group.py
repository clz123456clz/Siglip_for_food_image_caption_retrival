import os
import sys
import io
import tarfile
from enum import Enum
from collections import defaultdict

from PIL import Image, ImageDraw
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# 避免 FlashAttention 警告
os.environ["TRANSFORMERS_NO_FLASH_ATTN"] = "1"

# =========================
# Florence 模型部分 (和之前一样)
# =========================

class TaskType(str, Enum):
    OPEN_VOCAB_DETECTION = "<OPEN_VOCABULARY_DETECTION>"

def run_example(task_prompt: TaskType, image, text_input, model, processor, device):
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
    except:
        return {}

def get_watermark_mask(image, model, processor, device):
    # 简化的调用逻辑
    text_input = "watermark"
    parsed_answer = run_example(TaskType.OPEN_VOCAB_DETECTION, image, text_input, model, processor, device)
    
    image_width, image_height = image.size
    total_area = image_width * image_height
    found_watermark = False
    
    # 简单的 bbox 判定
    if "<OPEN_VOCABULARY_DETECTION>" in parsed_answer:
        bboxes = parsed_answer["<OPEN_VOCABULARY_DETECTION>"].get("bboxes", [])
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            if (x2 - x1) * (y2 - y1) <= 0.8 * total_area:
                found_watermark = True
                break # 只要找到一个合理的水印框就判死刑
    
    return found_watermark

# =========================
# 核心过滤逻辑：处理分组 (jpg+txt+json)
# =========================

VALID_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

def filter_tar_group(
    tar_path: str,
    model,
    processor,
    device: str,
    out_suffix: str = "_nowm.tar",
    max_side: int = 1024,
):
    if not os.path.exists(tar_path):
        print(f"[ERROR] Not found: {tar_path}")
        return

    base_dir = os.path.dirname(tar_path)
    base_name = os.path.basename(tar_path)
    out_path = os.path.join(base_dir, os.path.splitext(base_name)[0] + out_suffix)

    print("=" * 60)
    print(f"[Input ] {tar_path}")
    print(f"[Output] {out_path}")

    # --- 状态管理 ---
    # buffer[key] = [ (tarinfo, file_bytes), ... ]
    # 用来暂存那些“图片还没出来，无法决定去留”的 txt/json 文件
    buffer = defaultdict(list)
    
    # decisions[key] = True (保留) / False (丢弃)
    # 用来记录已经处理过的图片的决定
    decisions = {}

    kept_groups = 0
    dropped_groups = 0
    
    # 辅助函数：写入文件到 tar_out
    def write_to_tar(tar_out, info, data_bytes):
        # 必须重置 tarinfo 里的 offset 等信息，否则会报错
        info = tarfile.TarInfo(name=info.name)
        info.size = len(data_bytes)
        tar_out.addfile(info, io.BytesIO(data_bytes))

    with tarfile.open(tar_path, "r") as tar_in, \
         tarfile.open(out_path, "w") as tar_out:
        
        for member in tar_in:
            if not member.isfile():
                continue
            
            # 解析文件名和后缀
            # name: "000000001.jpg" -> key: "000000001", ext: ".jpg"
            file_name = os.path.basename(member.name)
            key, ext = os.path.splitext(file_name)
            ext = ext.lower()
            
            # 读取文件内容
            f = tar_in.extractfile(member)
            if f is None: continue
            content = f.read()

            # --- 情况 1: 是图片 ---
            if ext in VALID_IMG_EXTS:
                # 1. 立即进行检测
                try:
                    img = Image.open(io.BytesIO(content)).convert("RGB")
                    # Resize 提速
                    if max_side and max(img.size) > max_side:
                        ratio = max_side / max(img.size)
                        img_proc = img.resize((int(img.size[0]*ratio), int(img.size[1]*ratio)))
                    else:
                        img_proc = img
                    
                    is_watermark = get_watermark_mask(img_proc, model, processor, device)
                    keep = not is_watermark
                
                except Exception as e:
                    print(f"[Error] {member.name}: {e} -> Drop")
                    keep = False # 出错就扔
                
                # 2. 记录决定
                decisions[key] = keep
                
                if keep:
                    # 写入当前图片
                    write_to_tar(tar_out, member, content)
                    kept_groups += 1
                    print(f"[KEEP] {key} (No WM)")
                    
                    # 3. 检查是否有暂存在 buffer 里的附属文件 (txt/json)，如果有，全部写入并清空
                    if key in buffer:
                        for b_info, b_data in buffer[key]:
                            write_to_tar(tar_out, b_info, b_data)
                        del buffer[key]
                else:
                    dropped_groups += 1
                    print(f"[DROP] {key} (Watermark)")
                    # 如果不需要保留，buffer 里的东西也可以清掉了（省内存）
                    if key in buffer:
                        del buffer[key]

            # --- 情况 2: 是附属文件 (txt, json) ---
            else:
                # 检查是否已经对这个 key 做过决定了？
                if key in decisions:
                    if decisions[key] == True:
                        # 之前图片已经通过了，直接写入这个附属文件
                        write_to_tar(tar_out, member, content)
                    else:
                        # 之前图片被毙了，这个文件也扔掉
                        pass 
                else:
                    # 图片还没出现，先把这个文件暂存在 buffer 里
                    buffer[key].append((member, content))

    print(f"[Done] Kept groups: {kept_groups} | Dropped groups: {dropped_groups}")
    print("=" * 60)

def main():
    if len(sys.argv) < 2:
        print("Usage: python florence_filter_group.py file1.tar file2.tar ...")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Florence-2 on {device}...")
    
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True).to(device).eval()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    for tar_file in sys.argv[1:]:
        filter_tar_group(tar_file, model, processor, device)

if __name__ == "__main__":
    main()