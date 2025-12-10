import os
import numpy as np
import matplotlib.pyplot as plt

# ====== Config ======

REAL_STATS_PATH = "./visualization/real_physical_stats.npz"
SYNTH_STATS_PATH = "./visualization/synth_physical_stats.npz"

OUT_DIR = "./visualization"

# To avoid extremely long plotting time for huge datasets,
# we can optionally subsample per distribution.
MAX_POINTS_PER_DIST = 100000

RANDOM_SEED = 42


def load_stats(path):
    """
    Load a .npz stats file as a dict of numpy arrays.
    """
    data = np.load(path)
    stats = {k: data[k] for k in data.files}
    return stats


def maybe_subsample(arr, max_points, rng):
    """
    Optionally subsample a 1D array if it is longer than max_points.

    Args:
        arr: 1D numpy array.
        max_points: maximum number of elements to keep.
        rng: numpy Generator used for random sampling.

    Returns:
        A possibly subsampled 1D numpy array.
    """
    n = arr.shape[0]
    if max_points is not None and n > max_points:
        idx = rng.choice(n, size=max_points, replace=False)
        arr = arr[idx]
    return arr


def plot_hist_two_dists(
    real_values,
    synth_values,
    xlabel,
    title,
    outfile,
    bins=100,
    log_y=False,
):
    """
    Plot an overlaid histogram comparing real vs synthetic distributions.

    Args:
        real_values: 1D array of real-image statistics.
        synth_values: 1D array of synthetic-image statistics.
        xlabel: label for the x-axis.
        title: figure title.
        outfile: path to save the figure (PNG).
        bins: number of histogram bins.
        log_y: whether to use log scale on y-axis.
    """
    plt.figure(figsize=(8, 6))

    # Plot real data
    plt.hist(
        real_values,
        bins=bins,
        density=True,
        alpha=0.5,
        label="real",
    )

    # Plot synthetic data
    plt.hist(
        synth_values,
        bins=bins,
        density=True,
        alpha=0.5,
        label="synthetic",
    )

    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()

    if log_y:
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"[save] {outfile}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[load] Loading real stats from {REAL_STATS_PATH}")
    real_stats = load_stats(REAL_STATS_PATH)

    print(f"[load] Loading synthetic stats from {SYNTH_STATS_PATH}")
    synth_stats = load_stats(SYNTH_STATS_PATH)

    rng = np.random.default_rng(RANDOM_SEED)

    # ====== 1. Brightness mean (grayscale) ======
    real_brightness = maybe_subsample(
        real_stats["brightness_mean"],
        MAX_POINTS_PER_DIST,
        rng,
    )
    synth_brightness = maybe_subsample(
        synth_stats["brightness_mean"],
        MAX_POINTS_PER_DIST,
        rng,
    )

    plot_hist_two_dists(
        real_brightness,
        synth_brightness,
        xlabel="Grayscale brightness mean (0–255)",
        title="Brightness distribution (real vs synthetic)",
        outfile=os.path.join(OUT_DIR, "brightness_mean_hist.png"),
        bins=80,
        log_y=False,
    )

    # ====== 2. Saturation mean (HSV S channel) ======
    real_sat = maybe_subsample(
        real_stats["mean_S"],
        MAX_POINTS_PER_DIST,
        rng,
    )
    synth_sat = maybe_subsample(
        synth_stats["mean_S"],
        MAX_POINTS_PER_DIST,
        rng,
    )

    plot_hist_two_dists(
        real_sat,
        synth_sat,
        xlabel="Mean saturation (HSV S, 0–255)",
        title="Saturation distribution (real vs synthetic)",
        outfile=os.path.join(OUT_DIR, "saturation_mean_hist.png"),
        bins=80,
        log_y=False,
    )

    # ====== 3. Sharpness (Laplacian variance) ======
    real_lap = maybe_subsample(
        real_stats["laplacian_var"],
        MAX_POINTS_PER_DIST,
        rng,
    )
    synth_lap = maybe_subsample(
        synth_stats["laplacian_var"],
        MAX_POINTS_PER_DIST,
        rng,
    )

    # Laplacian variance can have a long tail, log-y sometimes helps
    plot_hist_two_dists(
        real_lap,
        synth_lap,
        xlabel="Laplacian variance (sharpness proxy)",
        title="Sharpness distribution (real vs synthetic)",
        outfile=os.path.join(OUT_DIR, "laplacian_var_hist.png"),
        bins=100,
        log_y=True,
    )

    # ====== 4. Edge strength (Sobel gradient mean) ======
    real_grad = maybe_subsample(
        real_stats["grad_mean"],
        MAX_POINTS_PER_DIST,
        rng,
    )
    synth_grad = maybe_subsample(
        synth_stats["grad_mean"],
        MAX_POINTS_PER_DIST,
        rng,
    )

    plot_hist_two_dists(
        real_grad,
        synth_grad,
        xlabel="Mean gradient magnitude (edge strength)",
        title="Edge strength distribution (real vs synthetic)",
        outfile=os.path.join(OUT_DIR, "grad_mean_hist.png"),
        bins=100,
        log_y=True,
    )

    print("[done] All comparison plots have been generated.")


if __name__ == "__main__":
    main()