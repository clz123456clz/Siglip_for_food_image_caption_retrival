import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ============================================================
# Config
# ============================================================

EMBED_DIR = "./visualization/embeddings"

TAGS = [
    "embeds_base",   
    "web_ft_e1",
    "web_ft_e2",
    "web_ft_e3",
    "web_ft_e4",
]

SAMPLES_PER_SPLIT = 1000
RANDOM_SEED = 42


def load_and_sample(path: str, max_samples: int, rng: np.random.Generator):
    """Load (N, D) array from .npy and randomly subsample <= max_samples rows."""
    arr = np.load(path)
    n = arr.shape[0]
    if n <= max_samples:
        return arr
    idx = rng.choice(n, size=max_samples, replace=False)
    return arr[idx]


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    all_embeddings = []
    all_domains = []   # "web" or "synth"
    all_epochs = []    # tag string

    # 1) Load all epochs
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
    all_domains_arr = np.array(all_domains)
    all_epochs_arr = np.array(all_epochs)

    print(f"[info] total points: {X.shape[0]}, dim: {X.shape[1]}")

    # 2) PCA -> 50D
    print("[pca] running PCA -> 50D")
    pca = PCA(n_components=50, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X)
    print("[pca] done")

    # 3) t-SNE -> 2D
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

    # 4) Plot, split by domain
    epoch_colors = {
        "embeds_base": "tab:blue",
        "web_ft_e1": "tab:orange",
        "web_ft_e2": "tab:green",
        "web_ft_e3": "tab:red",
        "web_ft_e4": "tab:purple",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    ax_web, ax_synth = axes

    # ---- web subfigure ----
    for tag in TAGS:
        mask = (all_domains_arr == "web") & (all_epochs_arr == tag)
        if not np.any(mask):
            continue
        pts = X_2d[mask]
        ax_web.scatter(
            pts[:, 0],
            pts[:, 1],
            s=8,
            alpha=0.6,
            c=epoch_colors[tag],
            label=tag,
        )
    ax_web.set_title("t-SNE (web images)")
    ax_web.legend(fontsize=8)

    # ---- synth subfigure ----
    for tag in TAGS:
        mask = (all_domains_arr == "synth") & (all_epochs_arr == tag)
        if not np.any(mask):
            continue
        pts = X_2d[mask]
        ax_synth.scatter(
            pts[:, 0],
            pts[:, 1],
            s=8,
            alpha=0.6,
            c=epoch_colors[tag],
            label=tag,
        )
    ax_synth.set_title("t-SNE (synthetic images)")
    ax_synth.legend(fontsize=8)

    fig.suptitle("t-SNE (PCA->50D) of web vs synthetic embeddings across epochs(web only)")
    plt.tight_layout()
    plt.savefig("./visualization/tsne_epochs_web_only_color_split.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()