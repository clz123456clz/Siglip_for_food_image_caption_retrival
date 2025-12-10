import tarfile
import os
from PIL import Image
from io import BytesIO

def extract_first_n_images(tar_path, output_dir, n=10):
    """
    Extract first n images from a tar file into output_dir
    and print each image's resolution (width, height).
    """

    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()

        count = 0
        for m in members:
            if not m.isfile():
                continue

            if not m.name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
                continue

            extracted = tar.extractfile(m)
            if extracted is None:
                continue

            img_bytes = extracted.read()
            try:
                img = Image.open(BytesIO(img_bytes))
            except Exception:
                continue

            width, height = img.size

            out_path = os.path.join(output_dir, os.path.basename(m.name))
            img.save(out_path)

            print(f"[Saved] {out_path}  |  Resolution: {width} x {height}")

            count += 1
            if count >= n:
                break

    print(f"\n✓ Done! Extracted {count} images to {output_dir}")

if __name__ == "__main__":
    tar_path = "/home/ubuntu/LLM-inference/liangzhao-project/vlm/Siglip_food/mtf2025_web_images_en/00000.tar"
    output_dir = "./sample_images"
    extract_first_n_images(tar_path, output_dir, n=10)
