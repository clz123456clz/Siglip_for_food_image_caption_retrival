import itertools
import os
from typing import Tuple, List, Optional
import shutil
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
from peft import PeftModel

__all__ = [
    "count_parameters",
    "save_checkpoint",
    "restore_checkpoint",
    "clear_checkpoint",
    "early_stopping",
    "train_epoch",
    "evaluate_epoch_siglip",
    "_list_epoch_dirs",
]

def count_parameters(model: torch.nn.Module) -> int:
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def restore_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str,
    cuda: bool = False,
    force: bool = False,
    pretrain: bool = False,
) -> Tuple[torch.nn.Module, int, list]:
    """
    Restore a model from checkpoints in `checkpoint_dir`.

    Behavior:
      • Lists available `epoch*` folders.
      • Interactive selection:
          - If user chooses 0  -> clear ALL checkpoints and return the (fresh) provided model.
          - If user chooses K>0:
                - If PEFT adapter files exist in that epoch folder -> rebuild PEFT model via
                  `PeftModel.from_pretrained(base_model, epoch_dir, is_trainable=True)`.
                - Else -> load classic `.pth.tar` weights into the provided model.

    Args:
        model: Base model (if restoring LoRA, this must be the backbone).
        checkpoint_dir: Directory containing epoch subfolders.
        cuda: Map to GPU if available.
        force: If True, forces interactive epoch selection even if none is available.
        pretrain: If True, allows partial loading (strict=False) for backbone pretraining.

    Returns:
        (restored_model, start_epoch, stats)
    """
    epochs = _list_epoch_dirs(checkpoint_dir) 
    if not epochs:
        print("No saved model checkpoints found.")
        if force:
            raise Exception("Checkpoint not found.")
        return model, 0, []

    valid_set = set(epochs)
    while True:
        print(f"Available epochs: {epochs}")
        print("Enter an epoch number to load, OR enter 0 to start from scratch (this will CLEAR all checkpoints).")
        try:
            inp_epoch = int(input(">> ").strip())
        except Exception:
            print("Invalid input. Please enter an integer.")
            continue

        if inp_epoch == 0:
            clear_checkpoint(checkpoint_dir, mode="all")
            print("✅ Starting from scratch. No checkpoint loaded.")
            return model, 0, []

        if inp_epoch in valid_set:
            break  # proceed to restore this epoch
        else:
            print(f"❌ Invalid epoch: {inp_epoch}. Please choose from {epochs}, or 0 to start from scratch.\n")
            
    # Restore from chosen epoch
    epoch_dir = os.path.join(checkpoint_dir, f"epoch{inp_epoch}")
    print(f"Loading from checkpoint: {epoch_dir}")

    # Detect PEFT adapter presence
    is_peft_epoch = (
        os.path.exists(os.path.join(epoch_dir, "adapter_config.json")) and
        os.path.exists(os.path.join(epoch_dir, "adapter_model.safetensors"))
    )

    stats = []
    start_epoch = inp_epoch

    if is_peft_epoch:
        # Rebuild PEFT model on top of the provided backbone
        restored = PeftModel.from_pretrained(
            model,
            epoch_dir,
            is_trainable=True
        )

        # Try to read stats/epoch
        meta_path = os.path.join(epoch_dir, "meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(
                meta_path,
                map_location=("cuda" if cuda and torch.cuda.is_available() else "cpu"),
            )
            stats = meta.get("stats", [])
            start_epoch = meta.get("epoch", inp_epoch)
        else:
            tstate = os.path.join(epoch_dir, "training_state.pt")
            if os.path.exists(tstate):
                ts = torch.load(
                    tstate,
                    map_location=("cuda" if cuda and torch.cuda.is_available() else "cpu"),
                )
                stats = ts.get("stats", [])
                start_epoch = ts.get("epoch", inp_epoch)

        print(f"=> Successfully restored LoRA adapter (epoch={start_epoch})")
        return restored, start_epoch, stats

    # Non-PEFT fallback
    ckpt_file = os.path.join(epoch_dir, "model.checkpoint.pth.tar")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Cannot find classic checkpoint at: {ckpt_file}")

    checkpoint = torch.load(
        ckpt_file,
        map_location=("cuda" if cuda and torch.cuda.is_available() else "cpu"),
        weights_only=False,
    )
    try:
        stats = checkpoint.get("stats", [])
        start_epoch = checkpoint.get("epoch", inp_epoch)
        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(f"=> Successfully restored classic checkpoint (epoch={start_epoch})")
    except Exception as e:
        print("=> Failed to restore checkpoint.")
        raise e

    return model, start_epoch, stats


def _list_epoch_dirs(checkpoint_dir: str) -> list[int]:
    """Return a sorted list of available epoch numbers under `checkpoint_dir`."""
    if not os.path.isdir(checkpoint_dir):
        return []
    epochs = []
    for d in os.listdir(checkpoint_dir):
        full = os.path.join(checkpoint_dir, d)
        if os.path.isdir(full) and d.startswith("epoch"):
            num = d[len("epoch"):]
            if num.isdigit():
                epochs.append(int(num))
    return sorted(epochs)

def clear_checkpoint(
    checkpoint_dir: str,
    mode: str = "all",
    epoch: Optional[int] = None,
) -> None:
    """
    Clear checkpoints.

    Args:
        checkpoint_dir: Root directory containing epoch subfolders.
        mode:
          - "all": remove ALL epoch folders.
          - "epoch": remove ONLY the specified epoch folder (requires `epoch`).
          - "keep_latest": remove all but the latest epoch.
        epoch: Epoch number to remove when mode="epoch".

    Notes:
        • This function is intentionally destructive. Use with care.
    """
    if not os.path.isdir(checkpoint_dir):
        print(f"[clear] Nothing to clear: {checkpoint_dir} does not exist.")
        return

    epochs = _list_epoch_dirs(checkpoint_dir)

    if mode == "all":
        for e in epochs:
            ep_dir = os.path.join(checkpoint_dir, f"epoch{e}")
            shutil.rmtree(ep_dir, ignore_errors=True)
        print(f"🧹 Cleared ALL checkpoints under: {checkpoint_dir}")

    elif mode == "epoch":
        if epoch is None:
            raise ValueError("When mode='epoch', you must provide an `epoch` to remove.")
        if epoch not in epochs:
            print(f"[clear] Epoch={epoch} not found. Nothing removed.")
            return
        ep_dir = os.path.join(checkpoint_dir, f"epoch{epoch}")
        shutil.rmtree(ep_dir, ignore_errors=True)
        print(f"🧹 Cleared checkpoint: {ep_dir}")

    elif mode == "keep_latest":
        if not epochs:
            print(f"[clear] No epochs found in: {checkpoint_dir}")
            return
        latest = epochs[-1]
        for e in epochs:
            if e == latest:
                continue
            ep_dir = os.path.join(checkpoint_dir, f"epoch{e}")
            shutil.rmtree(ep_dir, ignore_errors=True)
        print(f"🧹 Cleared all but the latest (epoch={latest}) in: {checkpoint_dir}")

    else:
        raise ValueError(f"Unknown clear mode: {mode}")



def early_stopping(stats: list, curr_count_to_patience: int, prev_val_loss: float) -> tuple[int, float]:
    """Calculate new patience and validation loss.

    Increment curr_count_to_patience by one if new loss is not less than prev_val_loss
    Otherwise, update prev_val_loss with the current val loss, and reset curr_count_to_patience to 0

    Returns: new values of curr_count_to_patience and prev_val_loss
    """
    new_val_loss = stats[-1]["val_loss"]
    if new_val_loss >= prev_val_loss:
        curr_count_to_patience+=1
    else:
        prev_val_loss = new_val_loss
        curr_count_to_patience = 0
    return curr_count_to_patience, prev_val_loss

def symmetric_siglip_loss_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, B] similarity matrix (before sigmoid), where the diagonal entries represent positive pairs.
    Returns: scalar value of the symmetric SigLIP (BCE-with-logits) loss.
    """
    B = logits.size(0)
    labels = torch.eye(B, device=logits.device, dtype=logits.dtype)
    loss_i2t = F.binary_cross_entropy_with_logits(logits,   labels)
    loss_t2i = F.binary_cross_entropy_with_logits(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)



def train_epoch(
    data_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    log_interval: int = 50,          # Print training logs every N steps
    max_steps: Optional[int] = None,    # Limit the number of steps per epoch (useful for IterableDatasets)
) -> None:
    """Train the model for one epoch, with a live tqdm progress bar."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    running = 0.0
    step = 0
    pbar = tqdm(data_loader, desc="train(batches)", unit="batch", dynamic_ncols=True)

    for X in pbar:
        if max_steps is not None and step >= max_steps:
            break

        optimizer.zero_grad()
        X = {k: (v.to(device, dtype=torch.bfloat16) if (k=="pixel_values" and torch.cuda.is_available()) else v.to(device))
             for k, v in X.items()}
        out = model(**X)
        logits = out.logits_per_image
        loss = symmetric_siglip_loss_from_logits(logits)

        loss.backward()
        optimizer.step()

        running = 0.9 * running + 0.1 * loss.item() if step > 0 else loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running:.4f}")

        if (step + 1) % log_interval == 0:
            tqdm.write(f"[train] step {step+1} | loss {loss.item():.4f} | avg {running:.4f}")

        step += 1

@torch.no_grad()
def evaluate_loss_only(data_loader, model, device="cuda", use_bf16=True):
    model.eval()
    running_loss, n_steps = 0.0, 0
    for batch in data_loader:
        batch = {
            k: (v.to(device, dtype=torch.bfloat16) if (k=="pixel_values" and use_bf16) else v.to(device))
            for k, v in batch.items()
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(**batch)
            logits = out.logits_per_image  # [B, B]
            loss = symmetric_siglip_loss_from_logits(logits)
        running_loss += loss.item()
        n_steps += 1
    model.train()
    return running_loss / max(n_steps, 1)



@torch.no_grad()
def compute_embeddings(loader, model, device="cuda", use_bf16=True):
    model.eval()
    img_feats, txt_feats = [], []
    for batch in tqdm(loader, desc="Encode val"):
        batch = {
            k: (v.to(device, dtype=torch.bfloat16) if (k=="pixel_values" and use_bf16) else v.to(device))
            for k, v in batch.items()
        }
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(**batch)
            v = out.image_embeds   # [B, D]
            t = out.text_embeds    # [B, D]

        # nomalize with cosine
        v = torch.nn.functional.normalize(v, dim=-1)
        t = torch.nn.functional.normalize(t, dim=-1)

        img_feats.append(v.float().cpu())
        txt_feats.append(t.float().cpu())

    img_feats = torch.cat(img_feats, dim=0)  # [N,D]
    txt_feats = torch.cat(txt_feats, dim=0)  # [N,D]
    model.train()
    return img_feats, txt_feats

def retrieval_metrics_from_embeddings(img_feats, txt_feats, ks=(1,5,10), sim_chunk=8192):
    """
    Compute retrieval metrics: Image-to-Text and Text-to-Image Recall@K, MRR, and Median Rank.
    The similarity matrix is computed in chunks to avoid allocating a full N*N matrix in memory.
    """
    N, D = img_feats.shape
    # calculate similarity：img @ txt^T（block-wise）
    sims = torch.empty((N, N), dtype=torch.float32)
    for start in range(0, N, sim_chunk):
        end = min(start + sim_chunk, N)
        sims[start:end] = img_feats[start:end] @ txt_feats.T  # [n_chunk, N]

    # ranks: i-j
    # i2t
    i2t_rank = torch.argsort(sims, dim=1, descending=True)
    # rank of correct text
    i2t_pos = (i2t_rank == torch.arange(N).unsqueeze(1)).nonzero()[:,1] + 1  # 1-based rank
    # t2i
    t2i_rank = torch.argsort(sims.T, dim=1, descending=True)
    t2i_pos = (t2i_rank == torch.arange(N).unsqueeze(1)).nonzero()[:,1] + 1

    def _recall_at(pos, k):  # pos: [N] ranks
        return float((pos <= k).float().mean().item())

    metrics = {}
    for k in ks:
        metrics[f"i2t_R@{k}"] = _recall_at(i2t_pos, k)
        metrics[f"t2i_R@{k}"] = _recall_at(t2i_pos, k)

    # MRR / Median
    metrics["i2t_MRR"] = float((1.0 / i2t_pos.float()).mean().item())
    metrics["t2i_MRR"] = float((1.0 / t2i_pos.float()).mean().item())
    metrics["i2t_MedR"] = float(i2t_pos.median().item())
    metrics["t2i_MedR"] = float(t2i_pos.median().item())
    return metrics


def evaluate_epoch_siglip(
    train_loader,
    val_loader,
    model,
    epoch:int,
    device="cuda",
    use_bf16=True,
    compute_retrieval=True,   
):
    train_loss = evaluate_loss_only(train_loader, model, device=device, use_bf16=use_bf16)
    val_loss   = evaluate_loss_only(val_loader,   model, device=device, use_bf16=use_bf16)

    stats = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}

    if compute_retrieval:
        img_feats, txt_feats = compute_embeddings(val_loader, model, device=device, use_bf16=use_bf16)
        r_metrics = retrieval_metrics_from_embeddings(img_feats, txt_feats, ks=(1,5,10))
        stats.update(r_metrics)

    return stats

def save_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    checkpoint_dir: str,
    stats: List[dict],
) -> None:
    """
    Save a PEFT (LoRA) model checkpoint along with training statistics.

    This function creates a folder named `epoch{N}` under `checkpoint_dir`,
    saves the LoRA adapter weights (`adapter_model.safetensors`) and config 
    (`adapter_config.json`) via `model.save_pretrained()`, and also stores a 
    lightweight `meta.pt` file containing the epoch number and accumulated 
    statistics.

    Args:
        model: The PEFT-wrapped model (i.e., a `PeftModel`).
        epoch: Current epoch number (integer).
        checkpoint_dir: Root directory for storing all epoch subfolders.
        stats: A list of dictionaries, each containing metrics for an epoch.

    Example:
        save_checkpoint(model, epoch=3, checkpoint_dir="siglip_lora_ckpt", stats=stats)
        # Produces:
        # siglip_lora_ckpt/
        # ├── epoch3/
        # │   ├── adapter_config.json
        # │   ├── adapter_model.safetensors
        # │   └── meta.pt
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_dir = os.path.join(checkpoint_dir, f"epoch{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    # ---- 1) Save PEFT adapter (automatically produces adapter_config.json + adapter_model.safetensors)
    try:
        model.save_pretrained(epoch_dir)
        print(f"[save] LoRA adapter saved at: {epoch_dir}")
    except Exception as e:
        print(f"[save] ⚠️ Failed to save adapter for epoch {epoch}: {e}")
        raise

    # ---- 2) Save training metadata (epoch number + stats list)
    meta = {"epoch": epoch, "stats": stats}
    meta_path = os.path.join(epoch_dir, "meta.pt")
    try:
        torch.save(meta, meta_path)
        print(f"[save] Meta info saved at: {meta_path}")
    except Exception as e:
        print(f"[save] ⚠️ Failed to save meta info: {e}")
        raise

    # ---- 3) Optional: clear older checkpoints if needed (you can call clear_checkpoint(..., mode='keep_latest'))
    print(f"✅ Checkpoint for epoch {epoch} successfully saved.")