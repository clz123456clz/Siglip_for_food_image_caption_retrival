import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import gc

from transformers import SiglipProcessor, SiglipModel
from peft import PeftModel


BASE_MODEL_NAME = "google/siglip-large-patch16-384"

LORA_ROOT_DIR = "./siglip_lora_V_l2_mixed05_qv_ckpt"

EPOCHS_TO_COMPARE = ["epoch1", "epoch5", "epoch10", "epoch17"] 

IMAGE_PATH = "./data/trap_watermark.jpg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_attention_heatmap(model, processor, image):
    """利用 CLS-Patch 相似度生成热力图"""
    model.eval()
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        if hasattr(model, "vision_model"):
            vision_module = model.vision_model
        elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
             vision_module = model.base_model.model.vision_model
        elif hasattr(model, "base_model") and hasattr(model.base_model, "vision_model"):
             vision_module = model.base_model.vision_model
        else:
            vision_module = model

        outputs = vision_module(inputs["pixel_values"], return_dict=True)

    hidden = outputs.last_hidden_state[0] 
    cls_token = hidden[0:1]      
    patch_tokens = hidden[1:]    


    scores = (patch_tokens @ cls_token.T).squeeze(-1)

    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()

    num_patches = patch_tokens.shape[0]
    grid_size = int(np.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        scores = scores[:grid_size*grid_size]

    attn_heatmap = scores.reshape(grid_size, grid_size).cpu().numpy()
    attn_heatmap_resized = cv2.resize(attn_heatmap, (image.size[0], image.size[1]))

    return attn_heatmap_resized

def overlay_heatmap(image, heatmap, alpha=0.6):
    img_np = np.array(image)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 找不到图片 {IMAGE_PATH}")
        return

    print(f"Loading processor: {BASE_MODEL_NAME}...")
    processor = SiglipProcessor.from_pretrained(BASE_MODEL_NAME)
    raw_image = Image.open(IMAGE_PATH).convert("RGB").resize((384, 384))

    num_plots = 2 + len(EPOCHS_TO_COMPARE)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    
    axes[0].imshow(raw_image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    print("\n[Base] Processing Base Model...")
    base_model = SiglipModel.from_pretrained(BASE_MODEL_NAME, attn_implementation="eager").to(DEVICE)
    
    heatmap_base = get_attention_heatmap(base_model, processor, raw_image)
    vis_base = overlay_heatmap(raw_image, heatmap_base)
    
    axes[1].imshow(vis_base)
    axes[1].set_title("Base Model\n(Pre-trained)", fontsize=14)
    axes[1].axis("off")

    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    for i, epoch_name in enumerate(EPOCHS_TO_COMPARE):
        plot_idx = i + 2
        ckpt_path = os.path.join(LORA_ROOT_DIR, epoch_name)
        
        if not os.path.exists(ckpt_path):
            print(f"⚠️ 跳过 {epoch_name}: 路径不存在")
            continue
            
        print(f"\n[{epoch_name}] Processing LoRA Checkpoint...")
        
        base_for_ft = SiglipModel.from_pretrained(BASE_MODEL_NAME, attn_implementation="eager")
        ft_model = PeftModel.from_pretrained(base_for_ft, ckpt_path).to(DEVICE)

        heatmap_ft = get_attention_heatmap(ft_model, processor, raw_image)
        vis_ft = overlay_heatmap(raw_image, heatmap_ft)
        
        axes[plot_idx].imshow(vis_ft)
        axes[plot_idx].set_title(f"Ours ({epoch_name})", fontsize=14)
        axes[plot_idx].axis("off")
        
        del ft_model
        torch.cuda.empty_cache()
        gc.collect()

    save_path = "./visualization/attention_evolution_2.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Result saved to: {save_path}")

if __name__ == "__main__":
    main()