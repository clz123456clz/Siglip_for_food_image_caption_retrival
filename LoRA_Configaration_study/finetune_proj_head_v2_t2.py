import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import nn
from transformers import AutoProcessor, SiglipModel
from peft import LoraConfig, get_peft_model, PeftModel
from dataset import get_mixed_train_val_loaders
from train import (
    evaluate_epoch_siglip,
    restore_checkpoint,
    train_epoch,
    early_stopping,
    save_checkpoint,
)
from utils import config, set_random_seed


def main():
    cfg = config("proj_head_en_v2_t2_qv")

    # =========================
    # 1. Setup & Base Model
    # =========================
    set_random_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = cfg["model_name"]
    model = SiglipModel.from_pretrained(model_name)
    model.to(device)
    model.train()

    # =========================
    # 2. Try to Restore from Checkpoint
    # =========================
    checkpoint_dir = cfg["checkpoint_dir"]
    try:
        # This automatically handles both PEFT and non-PEFT cases
        model, start_epoch, stats = restore_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            cuda=True,
            force=True,   # will prompt user to select epoch or 0 for fresh start
            pretrain=False,
        )
        print(f"✅ Model restored. Resume from epoch {start_epoch}.")
    except Exception as e:
        print(f"⚠️ No valid checkpoint found. Training from scratch.\n{e}")
        start_epoch, stats = 0, []

    # =========================
    # 3. Inject LoRA if Starting Fresh
    # =========================
    # 分开控制 Vision / Text encoder 调整的层数
    # 兼容老配置：如果没写就退回到 last_n_layers 或默认值
    V_LAST_N = int(
        cfg.get(
            "last_n_vision_layers",
            cfg.get("last_n_layers", 0)
        )
    )
    T_LAST_N = int(
        cfg.get(
            "last_n_text_layers",
            0  # 默认 text 不调
        )
    )

    # Linear submodules eligible for LoRA
    leaf_allow = {"q_proj", "v_proj"}

    # ---------------------------------------------------------
    # Freeze ALL parameters initially
    # ---------------------------------------------------------
    for n, p in model.named_parameters():
        p.requires_grad_(False)

    # ---------------------------------------------------------
    # Explicitly Unfreeze Heads and Logit Scale
    # ---------------------------------------------------------
    # 1. Logit Scale (usually a single trainable scalar)
    if hasattr(model, "logit_scale") and isinstance(model.logit_scale, nn.Parameter):
        model.logit_scale.requires_grad_(True)
        print("Logit Scale Unfrozen.")
    elif hasattr(model, "config") and hasattr(model.config, "logit_scale"):
        # 有些实现 logit_scale 在更深层级
        for name, param in model.named_parameters():
            if "logit_scale" in name:
                param.requires_grad_(True)
                print(f"Logit Scale Unfrozen via: {name}")

    # 2. Vision Head
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "head"):
        for n, p in model.vision_model.head.named_parameters():
            p.requires_grad_(True)
        print("Vision Head Unfrozen.")

    # 3. Text Head（有的话就放开）
    if hasattr(model, "text_model") and hasattr(model.text_model, "head"):
        for n, p in model.text_model.head.named_parameters():
            p.requires_grad_(True)
        print("Text Head Unfrozen.")

    # ---------------------------------------------------------
    # Helper: find the maximum encoder layer index
    # ---------------------------------------------------------
    def _max_layer_index(prefix: str) -> int:
        """Return the largest layer index under the given prefix (e.g., 'vision_model.encoder.layers.')."""
        max_idx = -1
        for name, _ in model.named_modules():
            if name.startswith(prefix):
                try:
                    idx = int(name.split(prefix, 1)[1].split(".")[0])
                    max_idx = max(max_idx, idx)
                except Exception:
                    pass
        return max_idx

    V_PREFIX = "vision_model.encoder.layers."
    T_PREFIX = "text_model.encoder.layers."

    v_max = _max_layer_index(V_PREFIX)
    t_max = _max_layer_index(T_PREFIX)

    if v_max < 0:
        raise RuntimeError("Cannot find vision encoder layers. Check module names in your SigLIP model.")
    if t_max < 0:
        print("⚠️ No text encoder layers found (or naming mismatch). Text LoRA will be skipped.")
        T_LAST_N = 0

    print(f"[Config] Vision last N layers for LoRA = {V_LAST_N}")
    print(f"[Config] Text   last N layers for LoRA = {T_LAST_N}")

    # ---------------------------------------------------------
    # Collect LoRA targets in the last N layers of BOTH encoders
    # ---------------------------------------------------------
    target_fullnames = []

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue

        leaf = name.split(".")[-1]
        if leaf not in leaf_allow:
            continue

        # Vision encoder layers
        if name.startswith(V_PREFIX) and V_LAST_N > 0:
            try:
                idx = int(name.split(V_PREFIX, 1)[1].split(".")[0])
            except Exception:
                idx = -1
            if idx >= max(0, v_max - V_LAST_N + 1):
                target_fullnames.append(name)

        # Text encoder layers
        if name.startswith(T_PREFIX) and T_LAST_N > 0:
            try:
                idx = int(name.split(T_PREFIX, 1)[1].split(".")[0])
            except Exception:
                idx = -1
            if idx >= max(0, t_max - T_LAST_N + 1):
                target_fullnames.append(name)

    target_fullnames = sorted(set(target_fullnames))

    print(
        f"[LoRA] Encoder-only symmetric targets: {len(target_fullnames)} "
        f"(vision last {V_LAST_N}, text last {T_LAST_N})"
    )
    for n in target_fullnames:
        print("   ", n)

    # ---------------------------------------------------------
    # Inject LoRA ONLY into the selected encoder layers
    # ---------------------------------------------------------
    lora_cfg = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        target_modules=target_fullnames,
    )
    model = get_peft_model(model, lora_cfg)

    # ---------------------------------------------------------
    # Sanity check: print total LoRA layers and trainable params
    # ---------------------------------------------------------
    from peft.tuners.lora import LoraLayer

    hits = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    print("LoRA layers =", len(hits))
    model.print_trainable_parameters()  # Should list only LoRA A/B params + heads/logit

    # bfloat16 训练
    model.to(device=device, dtype=torch.bfloat16)

    # =========================
    # 4. Optimizer & Dataloaders
    # =========================
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
        # Train model
        train_epoch(train_loader, model, optimizer)

        epoch_stat = evaluate_epoch_siglip(train_loader, val_loader, model, epoch)
        stats.append(epoch_stat)

        # save model parameters for current epoch
        save_checkpoint(model, epoch + 1, checkpoint_dir, stats)

        # update early stopping parameters
        curr_count_to_patience, prev_val_loss = early_stopping(
            stats, curr_count_to_patience, prev_val_loss
        )

        epoch += 1

    print("Finished Training")


if __name__ == "__main__":
    main()
