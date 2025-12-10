import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import nn

from transformers import AutoProcessor, SiglipModel
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer

from dataset import get_mixed_train_val_loaders
from train import (
    evaluate_epoch_siglip,
    restore_checkpoint,
    train_epoch,
    early_stopping,
    save_checkpoint,
)
from utils import config, set_random_seed


def configure_trainable_params(model: nn.Module) -> None:
    """
    Freeze backbone parameters and only keep LoRA + heads + logit_scale trainable.
    This function works for both:
      - freshly LoRA-injected SigLIP (get_peft_model)
      - restored PEFT model (PeftModel.from_pretrained)
    """
    # 1) Freeze everything first
    for name, p in model.named_parameters():
        p.requires_grad_(False)

    # Helper to unfreeze module parameters
    def _unfreeze_module(m: nn.Module, prefix: str):
        for n, p in m.named_parameters():
            p.requires_grad_(True)

    # 2) Unfreeze logit_scale (search by name; may be nested)
    for name, p in model.named_parameters():
        if "logit_scale" in name:
            p.requires_grad_(True)

    # 3) Unfreeze heads (vision / text), both on bare SigLIP and inside base_model
    #    because PeftModel wraps the original model under `base_model`.
    candidates = []

    # Bare SigLIP
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "head"):
        candidates.append(model.vision_model.head)
    if hasattr(model, "text_model") and hasattr(model.text_model, "head"):
        candidates.append(model.text_model.head)

    # Wrapped inside PEFT
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "vision_model") and hasattr(base.vision_model, "head"):
            candidates.append(base.vision_model.head)
        if hasattr(base, "text_model") and hasattr(base.text_model, "head"):
            candidates.append(base.text_model.head)

    for head in candidates:
        _unfreeze_module(head, "head")

    # 4) Unfreeze all LoRA parameters (wherever they live)
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            for p in module.parameters():
                p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[trainable] total trainable params: {n_trainable / 1e6:.2f}M")


def detect_prefix(model, mode="vision"):
    """
    Detect encoder layer prefix dynamically so that it works for:
      - bare SiglipModel: 'vision_model.encoder.layers.'
      - wrapped in PeftModel: 'base_model.vision_model.encoder.layers.'
    """
    keys = [name for name, _ in model.named_modules()]
    if mode == "vision":
        for k in keys:
            if "vision_model.encoder.layers" in k:
                return k.split("encoder.layers")[0] + "encoder.layers."
    else:
        for k in keys:
            if "text_model.encoder.layers" in k:
                return k.split("encoder.layers")[0] + "encoder.layers."
    return None


def _max_layer_index(model, prefix: str) -> int:
    """Return largest encoder layer index under given prefix."""
    max_idx = -1
    for name, _ in model.named_modules():
        if name.startswith(prefix):
            try:
                idx = int(name.split(prefix, 1)[1].split(".")[0])
                max_idx = max(max_idx, idx)
            except Exception:
                pass
    return max_idx


def main():
    cfg = config("proj_head_en_mixed")

    # ===================================
    # 1. Setup device & base SigLIP model
    # ===================================
    set_random_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg["model_name"]
    base_model = SiglipModel.from_pretrained(model_name)
    base_model.to(device)

    checkpoint_dir = cfg["checkpoint_dir"]

    # ===================================
    # 2. Try to restore from checkpoint
    # ===================================
    # If checkpoint exists and user chooses an epoch,
    # restore_checkpoint will return a PeftModel wrapping base_model.
    # If user chooses 0 or there is no checkpoint, we just get the original base_model.
    try:
        model, start_epoch, stats = restore_checkpoint(
            model=base_model,
            checkpoint_dir=checkpoint_dir,
            cuda=True,
            force=True,   # keep your original behavior: ask which epoch to load
            pretrain=False,
        )
        resumed = start_epoch > 0 and isinstance(model, (PeftModel,))
        if resumed:
            print(f"✅ Restored PEFT model from checkpoint. Resume from epoch {start_epoch}.")
        else:
            print("⚠️ No checkpoint selected / found. Start training from scratch (epoch 0).")
            start_epoch, stats = 0, []
            model = base_model  # ensure we are using the plain SigLIP model
    except Exception as e:
        # If restore completely fails (directory not found, etc.), we start from scratch.
        print(f"⚠️ Failed to restore checkpoint. Training from scratch.\n{e}")
        start_epoch, stats = 0, []
        model = base_model
        resumed = False

    # ===================================
    # 3. Inject LoRA ONLY when starting fresh
    # ===================================
    LAST_N = int(cfg["last_n_layers"])
    leaf_allow = {"q_proj", "v_proj", "out_proj"}

    if not resumed:
        # ----- determine encoder prefixes on bare SigLIP -----
        V_PREFIX = detect_prefix(model, mode="vision")
        T_PREFIX = detect_prefix(model, mode="text")

        if V_PREFIX is None:
            raise RuntimeError("Cannot find vision encoder layers in SigLIP model.")

        v_max = _max_layer_index(model, V_PREFIX)
        t_max = _max_layer_index(model, T_PREFIX) if T_PREFIX is not None else -1

        # ----- collect target full module names for LoRA -----
        target_fullnames = []
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            leaf = name.split(".")[-1]
            if leaf not in leaf_allow:
                continue

            # Vision encoder layers
            if V_PREFIX and name.startswith(V_PREFIX):
                idx = int(name.split(V_PREFIX, 1)[1].split(".")[0])
                if idx >= max(0, v_max - LAST_N + 1):
                    target_fullnames.append(name)

            # Text encoder layers (if present)
            if T_PREFIX and name.startswith(T_PREFIX):
                idx = int(name.split(T_PREFIX, 1)[1].split(".")[0])
                if idx >= max(0, t_max - LAST_N + 1):
                    target_fullnames.append(name)

        target_fullnames = sorted(set(target_fullnames))
        print(f"[LoRA] Encoder-only symmetric targets: {len(target_fullnames)} (last {LAST_N} layers)")
        for n in target_fullnames:
            print("   ", n)

        # ----- inject LoRA -----
        lora_cfg = LoraConfig(
            r=cfg["r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            bias="none",
            target_modules=target_fullnames,
        )
        model = get_peft_model(model, lora_cfg)
        print("✅ LoRA adapters injected into SigLIP encoder.")
    else:
        print("🔁 LoRA adapters already loaded from checkpoint; skip injection.")

    # ensure model is on device (after possible PEFT wrapping)
    model.to(device=device, dtype=torch.bfloat16)

    # ===================================
    # 4. Configure trainable params (LoRA + heads + logit_scale)
    # ===================================
    configure_trainable_params(model)
    if isinstance(model, PeftModel):
        # convenient PEFT summary
        model.print_trainable_parameters()

    # ===================================
    # 5. Optimizer & Dataloaders
    # ===================================
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    web_dir = "mtf2025_web_images_en"
    web_tars = [
        "00000.tar",
        "00001.tar",
        "00002.tar",
        "00003.tar",
        "00004.tar",
        "00005.tar",
        "00006.tar",
        "00007.tar",
        "00008.tar",
        "00009.tar",
        "00010.tar",
        "00011.tar",
        "00012.tar",
        "00013.tar",
    ]

    train_loader, val_loader = get_mixed_train_val_loaders(
        total_pairs=cfg["total_pairs"],
        tr_val_ratio=cfg["train_val_ratio"],
        web_synth_ratio=cfg["web_synth_ratio"],
        batch_size=cfg["batch_size"],
        web_tar_dir=web_dir,
        web_tar_names=web_tars,
        synth_hf_name="jesusmolrdv/MTF25-VLM-Challenge-Dataset-Synth",
        synth_split="train",
        num_workers=8,
        processor_name="google/siglip-large-patch16-384",
    )

    print(f"Training from epoch {start_epoch + 1} onward...")

    prev_val_loss = stats[-1]["val_loss"] if stats else float("inf")
    patience = cfg["patience"]
    curr_count_to_patience = 0

    epoch = start_epoch
    while curr_count_to_patience < patience:
        # ----- train one epoch -----
        train_epoch(train_loader, model, optimizer)

        # ----- evaluate & compute retrieval metrics -----
        epoch_stat = evaluate_epoch_siglip(train_loader, val_loader, model, epoch)
        stats.append(epoch_stat)

        # ----- save checkpoint for this epoch -----
        save_checkpoint(model, epoch + 1, checkpoint_dir, stats)

        # ----- early stopping update -----
        curr_count_to_patience, prev_val_loss = early_stopping(
            stats, curr_count_to_patience, prev_val_loss
        )

        epoch += 1

    print("Finished Training")


if __name__ == "__main__":
    main()