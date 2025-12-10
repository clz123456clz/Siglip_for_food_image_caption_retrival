import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from dataset import get_train_val_loaders
from train import evaluate_epoch_siglip, restore_checkpoint, train_epoch, early_stopping, save_checkpoint
from transformers import AutoProcessor, SiglipModel
from peft import LoraConfig, get_peft_model, PeftModel
from utils import config, set_random_seed
from torch import nn

def main():
    # =========================
    # 1. Setup & Base Model
    # =========================
    set_random_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = config("model_name")
    processor = AutoProcessor.from_pretrained(model_name)
    model = SiglipModel.from_pretrained(model_name)
    model.to(device)
    model.train()

    # =========================
    # 2. Try to Restore from Checkpoint
    # =========================
    checkpoint_dir = config("checkpoint_dir")  # e.g. "siglip_lora_vit_ckpt"
    try:
        # This automatically handles both PEFT and non-PEFT cases
        model, start_epoch, stats = restore_checkpoint(
            model=model,
            checkpoint_dir=checkpoint_dir,
            cuda=True,
            force=True,   # will prompt user to select epoch or 0 for fresh start
            pretrain=False
        )
        print(f"✅ Model restored. Resume from epoch {start_epoch}.")
    except Exception as e:
        print(f"⚠️ No valid checkpoint found. Training from scratch.\n{e}")
        start_epoch, stats = 0, []

    # =========================
    # 3. Inject LoRA if Starting Fresh
    # =========================
    # Number of last encoder layers to apply LoRA (default: 4 if not defined in config)
    LAST_N = int(config("last_n_layers"))
    # Linear submodules eligible for LoRA
    leaf_allow = {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}

    # ---------------------------------------------------------
    # Freeze ALL parameters, including heads and logit_scale
    # ---------------------------------------------------------
    for n, p in model.named_parameters():
        p.requires_grad_(False)

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
    if v_max < 0 or t_max < 0:
        raise RuntimeError("Cannot find encoder layers. Check module names in your SigLIP model.")

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
        if name.startswith(V_PREFIX):
            idx = int(name.split(V_PREFIX, 1)[1].split(".")[0])
            if idx >= max(0, v_max - LAST_N + 1):
                target_fullnames.append(name)

        # Text encoder layers
        elif name.startswith(T_PREFIX):
            idx = int(name.split(T_PREFIX, 1)[1].split(".")[0])
            if idx >= max(0, t_max - LAST_N + 1):
                target_fullnames.append(name)

    target_fullnames = sorted(set(target_fullnames))
    print(f"[LoRA] Encoder-only symmetric targets: {len(target_fullnames)} (last {LAST_N} layers)")
    for n in target_fullnames[:]:
        print("   ", n)

    # ---------------------------------------------------------
    # Inject LoRA ONLY into the selected encoder layers
    # ---------------------------------------------------------
    lora_cfg = LoraConfig(
        r=config("r"),
        lora_alpha=config("lora_alpha"),
        lora_dropout=config("lora_dropout"),
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
    model.print_trainable_parameters()  # Should list only LoRA A/B params (heads remain frozen)

    # =========================
    # 4. Optimizer & Dataloaders
    # =========================
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config("lr"),
        weight_decay=config("weight_decay"),
    )

    train_loader, val_loader = get_train_val_loaders(
        batch_size=config("batch_size"),
        tr_val_ratio=config("train_val_ratio"),  # e.g., 0.95 = 95% train / 5% val
    )

    print(f"Training from epoch {start_epoch + 1} onward...")

    prev_val_loss = stats[-1]["val_loss"] if stats else float("inf")

    patience = config("patience")
    curr_count_to_patience = 0

    epoch = start_epoch
    while curr_count_to_patience < patience:
        # Train model
        train_epoch(train_loader, model, optimizer)

        epoch_stat = evaluate_epoch_siglip(train_loader, val_loader, model, epoch)
        stats.append(epoch_stat)

        # save model parameters for current epoch
        save_checkpoint(model, epoch+1, checkpoint_dir, stats)

        # update early stopping parameters
        curr_count_to_patience, prev_val_loss = early_stopping(stats, curr_count_to_patience, prev_val_loss)

        epoch += 1

    print("Finished Training")

if __name__ == "__main__":
    main()