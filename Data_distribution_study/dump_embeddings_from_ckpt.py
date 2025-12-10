import os
import argparse
import numpy as np
import torch
from typing import Optional  

from transformers import AutoProcessor, SiglipModel

from dataset import get_train_val_loaders
from dataset_synth import build_synth_dataloaders

from train import restore_checkpoint
from utils import SEED, config

MAXREAL = 1000
MAXSYNTH = 1000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_image_embeddings(dataloader, model, max_samples: Optional[int] = None):
    """
    Iterate over a DataLoader and extract image embeddings.

    Args:
        dataloader: PyTorch DataLoader that yields batches with a
                    "pixel_values" key produced by a SigLIP processor.
        model:      SigLIP model (SiglipModel or PEFT-wrapped model)
                    that implements `get_image_features`.
        max_samples: If not None, stop after collecting at least this
                     many samples (useful for limiting runtime).

    Returns:
        A NumPy array of shape (N, D) containing image embeddings,
        L2-normalized along the feature dimension.
    """
    model.eval()
    all_embeds = []
    n_seen = 0

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(DEVICE)  # (B, C, H, W)

            # Forward through SigLIP vision tower
            image_embeds = model.get_image_features(pixel_values=pixel_values)  # (B, D)

            # Normalize embeddings for stability in similarity / visualization
            image_embeds = torch.nn.functional.normalize(image_embeds, dim=-1)

            all_embeds.append(image_embeds.cpu())
            n_seen += image_embeds.size(0)

            if max_samples is not None and n_seen >= max_samples:
                print(f"[info] Reached max_samples={max_samples}, stopping early.")
                break

    if not all_embeds:
        raise RuntimeError(
            "No embeddings extracted. Check your dataloader and batch keys."
        )

    all_embeds = torch.cat(all_embeds, dim=0)  # (N, D)
    if max_samples is not None and all_embeds.shape[0] > max_samples:
        all_embeds = all_embeds[:max_samples]

    print(f"[info] Final embedding shape: {all_embeds.shape}")
    return all_embeds.numpy()


def build_loaders_for_eval(
    processor_name: str,
    max_real_samples: Optional[int] = None,
    max_synth_samples: Optional[int] = None,
):
    """
    Construct web and synthetic dataloaders for evaluation.

    Returns:
        (web_loader, synth_loader)
    """
    print("\n[web] Building web train/val loaders ...")
    web_train_loader, web_val_loader = get_train_val_loaders(
        batch_size=64,
        tr_val_ratio=0.90,
        shards_glob="mtf2025_web_images_en/[0-9][0-9][0-9][0-9][0-9].tar",
        num_workers=4,
        processor_name=processor_name,
        shuffle_buffer=1000,
    )

    # Use the val split for analysis so it's independent of training batches
    web_loader = web_val_loader

    print("\n[synth] Building synthetic train/val loaders ...")
    synth_train_loader, synth_val_loader = build_synth_dataloaders(
        batch_size=64,
        tr_val_ratio=0.95,
        processor_name=processor_name,
        num_workers=4,
        shuffle_train=True,
        shuffle_val=False,
    )

    synth_loader = synth_val_loader

    return web_loader, synth_loader


def load_model_from_checkpoint(model_name: str, ckpt_dir: str):
    """
    Create a SigLIP model and restore weights from a checkpoint.

    Args:
        model_name:  Hugging Face model name (e.g. "google/siglip-large-patch16-384").
        ckpt_dir:    Directory where training checkpoints are stored.

    Returns:
        model:       A SiglipModel or PEFT-wrapped model moved to DEVICE.
        start_epoch: The epoch index returned by restore_checkpoint.
        stats:       Training statistics loaded from the checkpoint (if any).
    """
    print(f"[model] Loading base SigLIP: {model_name}")
    base_model = SiglipModel.from_pretrained(model_name)
    base_model.to(DEVICE)

    model, start_epoch, stats = restore_checkpoint(
        model=base_model,
        checkpoint_dir=ckpt_dir,
        cuda=True,
        force=True, 
        pretrain=False,
    )

    print(f"[ckpt] Restored epoch = {start_epoch}")
    model.to(DEVICE)
    return model, start_epoch, stats


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Dump SigLIP image embeddings from a specific checkpoint."
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag used in output filenames, e.g. 'synth_ft_e15'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = config("V_last2_qv_en_mixed") 


    # 1) Build evaluation dataloaders for web + synthetic images
    web_loader, synth_loader = build_loaders_for_eval(
        processor_name=cfg["model_name"],
        max_real_samples=MAXREAL,
        max_synth_samples=MAXSYNTH,
    )

    # 2) Load model weights from the specified checkpoint
    model, start_epoch, stats = load_model_from_checkpoint(
        model_name=cfg["model_name"],
        ckpt_dir=cfg["checkpoint_dir"],
    )
    print(f"[info] Using checkpoint epoch={start_epoch} for embedding dump.")

    out_dir = "./visualization/embeddings"
    os.makedirs(out_dir, exist_ok=True)

    # 3) Extract embeddings for web images
    print("\n[embed] Extracting WEB embeddings ...")
    web_embeds = extract_image_embeddings(
        dataloader=web_loader,
        model=model,
        max_samples=MAXREAL,
    )
    web_out_path = os.path.join(out_dir, f"web_{args.tag}.npy")
    np.save(web_out_path, web_embeds)
    print(f"[save] Web embeddings saved to: {web_out_path}")

    # 4) Extract embeddings for synthetic images
    print("\n[embed] Extracting SYNTH embeddings ...")
    synth_embeds = extract_image_embeddings(
        dataloader=synth_loader,
        model=model,
        max_samples=MAXSYNTH,
    )
    synth_out_path = os.path.join(out_dir, f"synth_{args.tag}.npy")
    np.save(synth_out_path, synth_embeds)
    print(f"[save] Synth embeddings saved to: {synth_out_path}")

    print("\n[done] All embeddings dumped.")


if __name__ == "__main__":
    main()