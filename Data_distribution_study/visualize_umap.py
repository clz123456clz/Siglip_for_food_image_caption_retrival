import numpy as np
import matplotlib.pyplot as plt
import umap

# ====== Config ======
WEB_EMBEDS_PATH = "./visualization/embeddings/web_embeds_base.npy"
SYN_EMBEDS_PATH = "./visualization/embeddings/web_nowm_embeds.npy"

# Number of samples from each group to visualize.
# UMAP can handle more points than t-SNE, but keep it reasonable.
N_WEB_SAMPLES = 5000
N_SYN_SAMPLES = 5000

RANDOM_SEED = 42


def load_and_sample(path, max_samples, rng):
    """
    Load a (N, D) embedding matrix from .npy and optionally subsample.

    Args:
        path: Path to the .npy file.
        max_samples: Maximum number of rows to keep. None = keep all.
        rng: NumPy random Generator used for subsampling.

    Returns:
        A NumPy array of shape (M, D), where M <= max_samples.
    """
    arr = np.load(path)  # (N, D)
    n = arr.shape[0]
    if max_samples is not None and n > max_samples:
        indices = rng.choice(n, size=max_samples, replace=False)
        arr = arr[indices]
    return arr


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # 1) Load and subsample real/synthetic embeddings
    print(f"[load] Loading web embeddings from {WEB_EMBEDS_PATH}")
    web = load_and_sample(WEB_EMBEDS_PATH, N_WEB_SAMPLES, rng)
    print(f"[load] Web shape after sampling: {web.shape}")

    print(f"[load] Loading web embeddings from {SYN_EMBEDS_PATH}")
    syn = load_and_sample(SYN_EMBEDS_PATH, N_SYN_SAMPLES, rng)
    print(f"[load] Web_nowm shape after sampling: {syn.shape}")

    # 2) Concatenate and build labels
    X = np.concatenate([web, syn], axis=0)  # (N_total, D)
    labels = np.concatenate(
        [
            np.zeros(web.shape[0], dtype=int),  # 0 = real (web)
            np.ones(syn.shape[0], dtype=int),   # 1 = synthetic (diffusion)
        ]
    )
    print(f"[info] Total points: {X.shape[0]}, dim: {X.shape[1]}")

    # 3) Run UMAP
    print("[umap] Running UMAP... this may take a bit")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=RANDOM_SEED,
    )
    X_2d = reducer.fit_transform(X)  # (N_total, 2)
    print("[umap] Done")

    # 4) Plot
    plt.figure(figsize=(8, 8))

    # Real web images
    plt.scatter(
        X_2d[labels == 0, 0],
        X_2d[labels == 0, 1],
        s=5,
        alpha=0.5,
        label="web (wm)",
    )

    # Synthetic diffusion images
    plt.scatter(
        X_2d[labels == 1, 0],
        X_2d[labels == 1, 1],
        s=5,
        alpha=0.5,
        label="web (nowm)",
    )

    plt.legend()
    plt.title("UMAP of image embeddings: real vs nowm")
    plt.tight_layout()
    plt.savefig("latent_umap_real_vs_nowm.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()