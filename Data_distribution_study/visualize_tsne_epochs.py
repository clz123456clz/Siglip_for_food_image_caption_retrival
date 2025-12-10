import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ============================================================
# Config
# ============================================================

EMBED_DIR = "./visualization/embeddings"

# Tags should match your file names:
#   web_{tag}.npy
#   synth_{tag}.npy
TAGS = [
    "embeds_base",   
    "mixed_ft_e5",
    "mixed_ft_e10",
    "mixed_ft_e15",
    "mixed_ft_e20",
]

# How many samples to draw per (domain, tag).
# t-SNE becomes slow if you use too many points.
SAMPLES_PER_SPLIT = 1000

RANDOM_SEED = 42


# ============================================================
# Helpers
# ============================================================

def load_and_sample(path: str, max_samples: int, rng: np.random.Generator):
    """
    Load a (N, D) embedding array from .npy and randomly subsample
    up to `max_samples` rows (without replacement).

    Args:
        path:        Path to the .npy file.
        max_samples: Maximum number of rows to keep.
        rng:         NumPy random generator.

    Returns:
        Array of shape (M, D), where M <= max_samples.
    """
    arr = np.load(path)  # (N, D)
    n = arr.shape[0]
    if n <= max_samples:
        return arr
    idx = rng.choice(n, size=max_samples, replace=False)
    return arr[idx]


# ============================================================
# Main
# ============================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    all_embeddings = []
    all_domains = [] 
    all_epochs = []  

    # -----------------------
    # 1) Load all epochs
    # -----------------------
    for tag in TAGS:
        web_path = f"{EMBED_DIR}/web_{tag}.npy"
        synth_path = f"{EMBED_DIR}/synth_{tag}.npy"

        print(f"[load] web:   {web_path}")
        web = load_and_sample(web_path, SAMPLES_PER_SPLIT, rng)
        print(f"        shape = {web.shape}")

        print(f"[load] synth: {synth_path}")
        synth = load_and_sample(synth_path, SAMPLES_PER_SPLIT, rng)
        print(f"        shape = {synth.shape}")

        all_embeddings.append(web)
        all_embeddings.append(synth)

        all_domains.extend(["web"] * web.shape[0])
        all_domains.extend(["synth"] * synth.shape[0])

        all_epochs.extend([tag] * (web.shape[0] + synth.shape[0]))

    X = np.concatenate(all_embeddings, axis=0)  # (N_total, D)
    all_domains = np.array(all_domains)
    all_epochs = np.array(all_epochs)

    print(f"[info] total points: {X.shape[0]}, dim: {X.shape[1]}")

    # -----------------------
    # 2) PCA -> 50D
    # -----------------------
    print("[pca] running PCA -> 50D")
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    print("[pca] done")

    # -----------------------
    # 3) t-SNE -> 2D
    # -----------------------
    print("[tsne] running t-SNE on PCA features...")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        learning_rate="auto",
        init="pca",
        metric="euclidean",
        random_state=RANDOM_SEED,
        n_iter=2000,
    )
    X_2d = tsne.fit_transform(X_pca)
    print("[tsne] done")

    # -----------------------
    # 4) Plot
    # -----------------------
    # color by domain, marker by epoch
    domain_to_color = {"web": "tab:blue", "synth": "tab:orange"}
    epoch_to_marker = {
        "embeds_base": "o",
        "synth_ft_e5": "^",
        "synth_ft_e10": "s",
        "synth_ft_e15": "D",
        "synth_ft_e20": "x",
    }

    plt.figure(figsize=(9, 9))

    for domain in ["web", "synth"]:
        for tag in TAGS:
            mask = (all_domains == domain) & (all_epochs == tag)
            if not np.any(mask):
                continue

            pts = X_2d[mask]

            plt.scatter(
                pts[:, 0],
                pts[:, 1],
                s=5,
                alpha=0.5,
                c=domain_to_color[domain],
                marker=epoch_to_marker.get(tag, "o"),
                label=f"{domain}-{tag}",
            )

    # de-duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), fontsize=8, loc="best")

    plt.title("t-SNE (PCA->50D) of image embeddings across epochs")
    plt.tight_layout()
    plt.savefig("./visualization/tsne_epochs_web_vs_synth.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()