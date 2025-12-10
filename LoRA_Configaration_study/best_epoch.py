from utils import config, set_random_seed
import torch
import os
from train import _list_epoch_dirs as list_epoch_dirs

cfg = config("V_last2_qv_en_mixed05")

def main():
    checkpoint_dir = cfg["checkpoint_dir"]
    epochs = list_epoch_dirs(checkpoint_dir)
    if not epochs:
        print("No saved model checkpoints found.")
    else:
        epoch = epochs[-1]
        epoch_dir = os.path.join(checkpoint_dir, f"epoch{epoch}")
        print(f"Loading from checkpoint: {epoch_dir}")
        meta_path = os.path.join(epoch_dir, "meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(
            meta_path,
            map_location=("cuda" if torch.cuda.is_available() else "cpu"),
        )
        stats = meta.get("stats", [])
        val_losses = [row["val_loss"] for row in stats] 
        best_epoch = min(range(len(val_losses)), key=lambda e: val_losses[e]) 


    print("\n" + "=" * 50)
    print(f"Best Epoch Summary")
    print("=" * 50)
    print(f"Best epoch (min val loss): {best_epoch}")
    print(f"Validation Loss : {stats[best_epoch]['val_loss']:.6f}\n")

    print(f"{'Metric':<15} | {'Value':>10}")
    print("-" * 30)
    print(f"{'i2t_MRR':<15} | {stats[best_epoch].get('i2t_MRR', float('nan')):>10.4f}")
    print(f"{'t2i_MRR':<15} | {stats[best_epoch].get('t2i_MRR', float('nan')):>10.4f}")
    print(f"{'i2t_MedR':<15} | {stats[best_epoch].get('i2t_MedR', float('nan')):>10.1f}")
    print(f"{'t2i_MedR':<15} | {stats[best_epoch].get('t2i_MedR', float('nan')):>10.1f}")
    print("-" * 30)

    ks = (1, 5, 10)
    print("\nRecall@K Metrics")
    print(f"{'Metric':<15} | {'Value':>10}")
    print("-" * 30)
    for k in ks:
        t2i_key = f"t2i_R@{k}"
        i2t_key = f"i2t_R@{k}"
        t2i_val = stats[best_epoch].get(t2i_key, float('nan'))
        i2t_val = stats[best_epoch].get(i2t_key, float('nan'))
        print(f"{i2t_key:<15} | {i2t_val:>10.4f}")
        print(f"{t2i_key:<15} | {t2i_val:>10.4f}")
    print("=" * 50)




if __name__ == "__main__":
    main()