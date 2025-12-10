import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ====== Config ======
WEB_EMBEDS_PATH = "./visualization/web_embeds.npy"
SYN_EMBEDS_PATH = "./visualization/synth_embeds.npy"

# Limit how many samples we use from each group.
# This keeps KNN training fast while still being representative.
N_WEB_SAMPLES = 20000
N_SYN_SAMPLES = 20000

K_FOR_CLASSIFIER = 5        # k for KNN classifier
K_FOR_NEIGHBORS_STATS = 5   # k for neighbor purity statistics

RANDOM_SEED = 42


def load_and_sample(path, max_samples, rng):
    """
    Load a (N, D) embedding matrix from .npy and optionally subsample rows.

    Args:
        path: Path to .npy file.
        max_samples: Max number of rows to keep. None = keep all.
        rng: np.random.Generator used to sample indices.

    Returns:
        A NumPy array of shape (M, D), where M <= max_samples.
    """
    arr = np.load(path)  # (N, D)
    n = arr.shape[0]
    if max_samples is not None and n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        arr = arr[idx]
    return arr


def knn_classifier_experiment(X, y, k):
    """
    Train a KNN classifier to distinguish real vs synthetic embeddings.

    Args:
        X: Feature matrix of shape (N, D).
        y: Labels of shape (N,) where 0 = real, 1 = synthetic.
        k: Number of neighbors for KNN.

    Prints:
        Classification accuracy, confusion matrix, and a short report.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y
    )

    # Note: metric="cosine" requires relatively recent scikit-learn.
    # If your version complains, switch to metric="minkowski" and
    # rely on the fact that embeddings are already normalized.
    clf = KNeighborsClassifier(
        n_neighbors=k,
        metric="cosine"
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("=== KNN classifier results ===")
    print(f"k = {k}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rows = true, cols = pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["real", "synthetic"]))
    print("==============================\n")


def neighbor_purity_stats(X, y, k):
    """
    Compute neighbor purity statistics:
      - For each point, find its k nearest neighbors.
      - For each class, report the average fraction of neighbors
        that belong to the same class (excluding the point itself).

    Args:
        X: Feature matrix of shape (N, D).
        y: Labels of shape (N,) where 0 = real, 1 = synthetic.
        k: Number of neighbors to consider.

    Prints:
        Average same-class neighbor ratio for real and synthetic points.
    """
    # Use cosine distance in neighbor search.
    nn = NearestNeighbors(
        n_neighbors=k + 1,  # +1 to account for the point itself at distance 0
        metric="cosine"
    )
    nn.fit(X)

    # indices: shape (N, k+1)
    distances, indices = nn.kneighbors(X, return_distance=True)

    # For each point, drop the first neighbor (itself)
    neighbor_indices = indices[:, 1:]  # (N, k)
    neighbor_labels = y[neighbor_indices]  # (N, k)

    # Expand y to shape (N, 1) for broadcasting
    y_expanded = y[:, None]
    same_class = (neighbor_labels == y_expanded).astype(np.float32)  # (N, k)

    # Average over neighbors
    same_class_ratio = same_class.mean(axis=1)  # (N,)

    # Compute averages for real vs synthetic
    real_mask = (y == 0)
    synth_mask = (y == 1)

    real_ratio = same_class_ratio[real_mask].mean()
    synth_ratio = same_class_ratio[synth_mask].mean()

    print("=== Neighbor purity statistics ===")
    print(f"k = {k}")
    print(f"Average same-class neighbor ratio (real):      {real_ratio:.4f}")
    print(f"Average same-class neighbor ratio (synthetic): {synth_ratio:.4f}")
    print("===================================\n")


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # 1) Load and subsample embeddings
    print(f"[load] Loading web embeddings from {WEB_EMBEDS_PATH}")
    web = load_and_sample(WEB_EMBEDS_PATH, N_WEB_SAMPLES, rng)
    print(f"[load] Web shape after sampling: {web.shape}")

    print(f"[load] Loading synthetic embeddings from {SYN_EMBEDS_PATH}")
    syn = load_and_sample(SYN_EMBEDS_PATH, N_SYN_SAMPLES, rng)
    print(f"[load] Synth shape after sampling: {syn.shape}")

    # 2) Build feature matrix and labels
    X = np.concatenate([web, syn], axis=0)  # (N_total, D)
    y = np.concatenate(
        [
            np.zeros(web.shape[0], dtype=int),  # 0 = real
            np.ones(syn.shape[0], dtype=int),   # 1 = synthetic
        ]
    )

    print(f"[info] Total points: {X.shape[0]}, dim: {X.shape[1]}")
    print(f"[info] Label counts: real={np.sum(y==0)}, synthetic={np.sum(y==1)}\n")

    # 3) Run KNN classifier experiment
    knn_classifier_experiment(X, y, K_FOR_CLASSIFIER)

    # 4) Run neighbor purity analysis
    neighbor_purity_stats(X, y, K_FOR_NEIGHBORS_STATS)


if __name__ == "__main__":
    main()