import os
import io
import tarfile
from PIL import Image, ImageDraw
from craft_text_detector import Craft


VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

def draw_craft_boxes(img, boxes, save_path):
    """Draw text boxes detected by CRAFT and save image."""
    draw = ImageDraw.Draw(img)
    for poly in boxes:
        pts = [(poly[i][0], poly[i][1]) for i in range(4)]
        draw.polygon(pts, outline="red", width=3)

    img.save(save_path, "JPEG", quality=95)
    print(f"[SAVE] {save_path}")

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preview_tar_craft.py 00000.tar")
        return

    tar_path = sys.argv[1]
    if not os.path.exists(tar_path):
        print(f"[ERROR] {tar_path} not found")
        return

    # Output folder
    out_dir = "./tar_preview"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Saving previews to: {out_dir}")

    # Load CRAFT
    device = "cuda" if torch.cuda.is_available() else "cpu"
    craft = Craft(output_dir=None, crop_type="box", cuda=(device == "cuda"))

    count = 0

    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if count >= 20:
                break

            if not member.isfile():
                continue

            if not member.name.lower().endswith(VALID_EXTS):
                continue

            f = tar.extractfile(member)
            if f is None:
                continue

            img_bytes = f.read()

            try:
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except:
                continue

            # CRAFT detection
            prediction = craft.detect_text(img)
            boxes = prediction.get("boxes", [])

            # Save visualization
            base = os.path.basename(member.name)
            name_no_ext, _ = os.path.splitext(base)
            save_path = os.path.join(out_dir, f"{name_no_ext}_bbox.jpeg")

            draw_craft_boxes(img, boxes, save_path)

            count += 1

    craft.unload_craftnet_model()
    craft.unload_refinenet_model()

    print(f"[DONE] Extracted and visualized {count} images.")

if __name__ == "__main__":
    import torch
    main()
