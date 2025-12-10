import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 尝试导入自动调整文字的库
try:
    from adjustText import adjust_text
except ImportError:
    print("❌ 错误: 请先安装 adjustText 库！")
    print("   运行命令: pip install adjustText")
    exit(1)

# ============================================================
# 1. 配置区域 (Configuration)
# ============================================================

EMBED_DIR = "./visualization/embeddings"
OUTPUT_DIR = "./visualization"
OUTPUT_BASENAME = "trajectory_tsne_web_vs_synth_adjusted"

# ⚠️ 关键：请把这里换成你实际存在的 Tag，并按时间顺序排列
TAGS = [
    "embeds_base",      # 起点 (Epoch 0)
    "web_ft_e1",        # 中间过程
    "web_ft_e2",
    "web_ft_e3",
    "web_ft_e4",
    # "clean_web_ft_e5" # 你的实验结果 (请修改为实际文件名)
]

# 采样数量
SAMPLES_PER_SPLIT = 1000 
RANDOM_SEED = 42

# 配色
COLORS = {
    "web": "#1f77b4",   # 蓝
    "synth": "#ff7f0e"  # 橙
}

# ============================================================
# 2. 辅助函数
# ============================================================

def load_and_sample(path, max_samples, rng):
    if not os.path.exists(path):
        print(f"[Warn] File not found: {path}")
        return None
    arr = np.load(path)
    n = arr.shape[0]
    if n <= max_samples:
        return arr
    idx = rng.choice(n, size=max_samples, replace=False)
    return arr[idx]

def draw_arrow(x1, y1, x2, y2, ax, color):
    """画箭头"""
    ax.annotate(
        "", 
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5, shrinkA=0, shrinkB=0)
    )

# ============================================================
# 3. 主程序
# ============================================================

def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # --- A: 加载数据 ---
    all_embeddings = []
    metadata = [] # (Domain, Tag)

    print("[1/4] Loading Data...")
    for tag in TAGS:
        web_path = f"{EMBED_DIR}/web_{tag}.npy"
        synth_path = f"{EMBED_DIR}/synth_{tag}.npy"

        web = load_and_sample(web_path, SAMPLES_PER_SPLIT, rng)
        synth = load_and_sample(synth_path, SAMPLES_PER_SPLIT, rng)

        if web is None or synth is None: continue

        all_embeddings.append(web)
        all_embeddings.append(synth)
        metadata.extend([("web", tag)] * len(web))
        metadata.extend([("synth", tag)] * len(synth))

    if not all_embeddings:
        print("[Error] No embeddings loaded.")
        return

    X = np.concatenate(all_embeddings, axis=0)
    meta_domain = np.array([m[0] for m in metadata])
    meta_tag = np.array([m[1] for m in metadata])

    # --- B: 降维 ---
    print(f"[2/4] Running PCA (768 -> 50)...")
    X_pca = PCA(n_components=50, random_state=RANDOM_SEED).fit_transform(X)

    print("[3/4] Running t-SNE (50 -> 2)...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=RANDOM_SEED, init='pca')
    X_2d = tsne.fit_transform(X_pca)

    # --- C: 画图 ---
    print("[4/4] Plotting Trajectory...")
    fig, ax = plt.subplots(figsize=(12, 10)) # 稍微加大一点画布，给文字留空间

    # 1. 画背景散点
    for dom in ["web", "synth"]:
        mask = (meta_domain == dom)
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1], 
            c=COLORS[dom], s=10, alpha=0.08, edgecolors='none',
            label=f"{dom.capitalize()} Distribution"
        )

    # 列表用于收集所有的文字对象，最后统一调整
    texts_to_adjust = []

    # 2. 画轨迹
    for dom in ["web", "synth"]:
        centroids_x = []
        centroids_y = []
        valid_tags = []

        for tag in TAGS:
            mask = (meta_domain == dom) & (meta_tag == tag)
            if np.any(mask):
                points = X_2d[mask]
                center = np.mean(points, axis=0)
                centroids_x.append(center[0])
                centroids_y.append(center[1])
                valid_tags.append(tag)
        
        if not centroids_x: continue

        # 连线
        ax.plot(
            centroids_x, centroids_y, 
            c=COLORS[dom], lw=2.5, linestyle='-', zorder=3,
            label=f"{dom.capitalize()} Trajectory"
        )
        
        # 标注关键点和箭头
        for i in range(len(centroids_x)):
            cx = centroids_x[i]
            cy = centroids_y[i]
            curr_tag = valid_tags[i]
            
            # 实心点
            ax.scatter(cx, cy, c=COLORS[dom], s=60, edgecolors='white', zorder=4)

            # 标签处理
            label_text = curr_tag.split('_')[-1]
            if "base" in curr_tag: label_text = "Base"
            
            # 🔥 关键修改：不再硬编码位置，而是创建 text 对象并收集起来
            # 我们初始把文字放在点的正上方，adjust_text 会负责把它移开
            t = ax.text(
                cx, cy, label_text, 
                fontsize=11, fontweight='bold', color=COLORS[dom], zorder=5
            )
            texts_to_adjust.append(t)

            # 箭头
            if i < len(centroids_x) - 1:
                next_x, next_y = centroids_x[i+1], centroids_y[i+1]
                mid_x = (cx + next_x) / 2
                mid_y = (cy + next_y) / 2
                draw_arrow(cx, cy, mid_x, mid_y, ax, COLORS[dom])

    # --- D: 自动调整文字位置 ---
    print("Auto-adjusting text labels (this might take a few seconds)...")
    adjust_text(
        texts_to_adjust,
        # 允许文字把点推开一点
        expand_points=(1.2, 1.2),
        # 如果移动了文字，用灰色细线连接到点
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        ax=ax
    )

    # 装饰
    ax.set_title("Feature Space Evolution: Web vs Synthetic", fontsize=16)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}.png")
    plt.savefig(output_path, dpi=300)
    print(f"[Done] Saved to {output_path}")

if __name__ == "__main__":
    main()