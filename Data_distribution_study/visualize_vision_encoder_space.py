import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ========== Config ==========
WEB_EMBEDS_PATH = "./visualization/embeddings/web_embeds_base.npy"
SYN_EMBEDS_PATH = "./visualization/embeddings/web_nowm_embeds.npy"

# How many samples to visualize from each set.
# t-SNE gets slow if you feed it too many points.
N_WEB_SAMPLES = 5000
N_SYN_SAMPLES = 5000
RANDOM_SEED = 42


def load_and_sample(path, max_samples, rng):
    """
    Load a (N, D) embedding array from .npy and optionally subsample.
    """
    arr = np.load(path)  # shape: (N, D)
    n = arr.shape[0]
    if max_samples is not None and n > max_samples:
        indices = rng.choice(n, size=max_samples, replace=False)
        arr = arr[indices]
    return arr


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # 1) Load and subsample embeddings
    print(f"[load] loading web embeddings from {WEB_EMBEDS_PATH}")
    web = load_and_sample(WEB_EMBEDS_PATH, N_WEB_SAMPLES, rng)
    print(f"[load] web shape after sampling: {web.shape}")

    print(f"[load] loading synthetic embeddings from {SYN_EMBEDS_PATH}")
    syn = load_and_sample(SYN_EMBEDS_PATH, N_SYN_SAMPLES, rng)
    print(f"[load] synth shape after sampling: {syn.shape}")

    # 2) Concatenate for joint visualization
    X = np.concatenate([web, syn], axis=0)  # shape: (N_web + N_syn, D)
    labels = np.concatenate([
        np.zeros(web.shape[0], dtype=int),   # 0 = real web images
        np.ones(syn.shape[0], dtype=int),    # 1 = synthetic images
    ])
    print(f"[info] total points: {X.shape[0]}, dim: {X.shape[1]}")

    # 3) Run t-SNE (you can switch to UMAP if you prefer)
    print("[tsne] running t-SNE... this may take a while")
    tsne = TSNE(
        n_components=2,
        metric="cosine",
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_SEED,
    )
    X_2d = tsne.fit_transform(X)  # shape: (N, 2)
    print("[tsne] done")

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

    # Synthetic images
    plt.scatter(
        X_2d[labels == 1, 0],
        X_2d[labels == 1, 1],
        s=5,
        alpha=0.5,
        label="web (nowm)",
    )

    plt.legend()
    plt.title("t-SNE of image embeddings: real vs nowm")
    plt.tight_layout()
    plt.savefig("latent_tsne_real_vs_nowm.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()