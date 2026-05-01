"""
GAN Synthesis Quality Evaluation
=================================
Checks whether your synthetic EEG STFT samples are realistic using:

  1. Per-class statistics     — mean / std comparison (real vs synthetic)
  2. RMSE                     — average pointwise distance (lower = better, paper target: ~0.21)
  3. Fréchet Distance (FD)    — distribution-level similarity (lower = better, paper target: ~0.75)
  4. PSD comparison plot      — visual frequency content check
  5. t-SNE plot               — cluster overlap (real vs fake should mix)
  6. TSTR accuracy            — Train-Synthetic Test-Real downstream check

Run after synthesize_gan.py has produced X_synthetic.pt and y_synthetic.pt.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.linalg import sqrtm
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_DIR = Path("../data/processed").resolve()

# ============================================================
# LOAD DATA
# ============================================================
def load_all():
    X_real = torch.load(DATA_DIR / "X_train.pt").numpy()          # (N_real, T, C, F)
    y_real = torch.load(DATA_DIR / "y_train.pt").numpy()

    X_syn  = torch.load(DATA_DIR / "X_synthetic.pt").numpy()      # (N_syn, T, C, F)
    y_syn  = torch.load(DATA_DIR / "y_synthetic.pt").numpy()

    X_test = torch.load(DATA_DIR / "X_test.pt").numpy()
    y_test = torch.load(DATA_DIR / "y_test.pt").numpy()

    print(f"Real train   : {X_real.shape}   labels: {np.bincount(y_real)}")
    print(f"Synthetic    : {X_syn.shape}    labels: {np.bincount(y_syn)}")
    print(f"Test (real)  : {X_test.shape}   labels: {np.bincount(y_test)}")

    return X_real, y_real, X_syn, y_syn, X_test, y_test


# ============================================================
# 1. PER-CLASS STATISTICS
# ============================================================
def check_statistics(X_real, y_real, X_syn, y_syn):
    print("\n" + "="*55)
    print("  1. PER-CLASS STATISTICS")
    print("="*55)
    label_names = ["low", "mid", "high"]

    for cls in range(3):
        r = X_real[y_real == cls]
        s = X_syn [y_syn  == cls]

        r_mean, r_std = r.mean(), r.std()
        s_mean, s_std = s.mean(), s.std()

        mean_diff = abs(r_mean - s_mean)
        std_diff  = abs(r_std  - s_std)

        status = "✔ GOOD" if mean_diff < 0.1 and std_diff < 0.1 else "⚠ CHECK"
        print(f"\n  Class '{label_names[cls]}'  ({status})")
        print(f"    Real  — mean: {r_mean:.4f}   std: {r_std:.4f}")
        print(f"    Synth — mean: {s_mean:.4f}   std: {s_std:.4f}")
        print(f"    Δmean: {mean_diff:.4f}   Δstd: {std_diff:.4f}")


# ============================================================
# 2. RMSE  (per class, then average)
# ============================================================
def compute_rmse(X_real, y_real, X_syn, y_syn):
    print("\n" + "="*55)
    print("  2. RMSE  (target ≈ 0.21 for LC-WGAN-GP)")
    print("="*55)
    label_names = ["low", "mid", "high"]
    rmse_list   = []

    for cls in range(3):
        r = X_real[y_real == cls]
        s = X_syn [y_syn  == cls]

        # Compare class-level mean spectrograms
        n = min(len(r), len(s))
        r_mean = r[:n].mean(axis=0)
        s_mean = s[:n].mean(axis=0)

        rmse = np.sqrt(((r_mean - s_mean) ** 2).mean())
        rmse_list.append(rmse)

        quality = "✔ Excellent" if rmse < 0.25 else ("OK" if rmse < 0.50 else "⚠ Poor")
        print(f"  Class '{label_names[cls]}' RMSE: {rmse:.4f}   {quality}")

    avg_rmse = np.mean(rmse_list)
    print(f"\n  Average RMSE: {avg_rmse:.4f}   (paper best: 0.21)")
    return avg_rmse


# ============================================================
# 3. FRÉCHET DISTANCE
# ============================================================
def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff  = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fd)


def compute_fd(X_real, y_real, X_syn, y_syn):
    print("\n" + "="*55)
    print("  3. FRÉCHET DISTANCE  (target ≈ 0.75 for LC-WGAN-GP)")
    print("="*55)
    label_names = ["low", "mid", "high"]
    fd_list     = []

    for cls in range(3):
        r = X_real[y_real == cls].reshape(sum(y_real == cls), -1)   # flatten
        s = X_syn [y_syn  == cls].reshape(sum(y_syn  == cls), -1)

        # PCA to 64 dims (FD is unstable in very high dims)
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(64, r.shape[0] - 1, s.shape[0] - 1))
        pca.fit(r)
        r_p = pca.transform(r)
        s_p = pca.transform(s)

        mu_r, sigma_r = r_p.mean(axis=0), np.cov(r_p, rowvar=False)
        mu_s, sigma_s = s_p.mean(axis=0), np.cov(s_p, rowvar=False)

        fd = frechet_distance(mu_r, sigma_r, mu_s, sigma_s)
        fd_list.append(fd)

        quality = "✔ Excellent" if fd < 1.0 else ("OK" if fd < 3.0 else "⚠ Poor")
        print(f"  Class '{label_names[cls]}' FD: {fd:.4f}   {quality}")

    avg_fd = np.mean(fd_list)
    print(f"\n  Average FD: {avg_fd:.4f}   (paper best: 0.75)")
    return avg_fd


# ============================================================
# 4. PSD COMPARISON PLOT
# ============================================================
def plot_psd(X_real, y_real, X_syn, y_syn, save_path):
    """
    Mean power across frequency axis (F dim) for each class.
    Real and synthetic curves should overlap closely.
    """
    label_names = ["Low Stress", "Mid Stress", "High Stress"]
    colors_r    = ["#2196F3", "#4CAF50", "#F44336"]
    colors_s    = ["#90CAF9", "#A5D6A7", "#EF9A9A"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), facecolor="#0D1117")
    fig.suptitle("PSD Comparison: Real vs Synthetic", color="white",
                 fontsize=14, fontweight="bold", y=1.02)

    for cls, ax in enumerate(axes):
        r = X_real[y_real == cls]   # (N, T, C, F)
        s = X_syn [y_syn  == cls]

        # Mean over samples, time, channels → power per freq bin
        r_psd = r.mean(axis=(0, 1, 2))   # (F,)
        s_psd = s.mean(axis=(0, 1, 2))

        freq_idx = np.arange(len(r_psd))

        ax.plot(freq_idx, r_psd, color=colors_r[cls], linewidth=2,
                label="Real",  alpha=0.9)
        ax.plot(freq_idx, s_psd, color=colors_s[cls], linewidth=2,
                label="Synth", alpha=0.9, linestyle="--")
        ax.fill_between(freq_idx, r_psd, s_psd,
                        alpha=0.15, color=colors_r[cls])

        ax.set_title(label_names[cls], color="white", fontsize=11)
        ax.set_xlabel("Frequency bin", color="#AAAAAA", fontsize=9)
        ax.set_ylabel("Mean power",    color="#AAAAAA", fontsize=9)
        ax.tick_params(colors="#AAAAAA")
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.legend(fontsize=8, facecolor="#1C2128", labelcolor="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="#0D1117")
    plt.close()
    print(f"\n  PSD plot saved → {save_path}")


# ============================================================
# 5. t-SNE PLOT
# ============================================================
def plot_tsne(X_real, y_real, X_syn, y_syn, save_path, n_samples=300):
    """
    Good synthesis: real and synthetic points of the same class
    should OVERLAP, not form separate islands.
    """
    # Subsample for speed
    def subsample(X, y, n):
        idx = np.random.choice(len(X), min(n, len(X)), replace=False)
        return X[idx], y[idx]

    Xr, yr = subsample(X_real, y_real, n_samples)
    Xs, ys = subsample(X_syn,  y_syn,  n_samples)

    X_all  = np.concatenate([Xr, Xs], axis=0).reshape(len(Xr) + len(Xs), -1)
    labels = np.concatenate([yr, ys + 3])   # 0,1,2 = real;  3,4,5 = synthetic

    # Reduce dims before t-SNE
    from sklearn.decomposition import PCA
    pca   = PCA(n_components=min(50, X_all.shape[1]))
    X_pca = pca.fit_transform(X_all)

    tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d   = tsne.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0D1117")
    class_names = ["Low", "Mid", "High"]
    markers_r   = ["o", "s", "^"]
    markers_s   = ["o", "s", "^"]
    colors_r    = ["#2196F3", "#4CAF50", "#F44336"]
    colors_s    = ["#90CAF9", "#A5D6A7", "#EF9A9A"]

    nr = len(Xr)
    for cls in range(3):
        mask_r = (labels[:nr]  == cls)
        mask_s = (labels[nr:] == cls + 3)

        ax.scatter(X_2d[:nr][mask_r, 0], X_2d[:nr][mask_r, 1],
                   c=colors_r[cls], marker=markers_r[cls],
                   s=40, alpha=0.7, label=f"Real  {class_names[cls]}")
        ax.scatter(X_2d[nr:][mask_s, 0], X_2d[nr:][mask_s, 1],
                   c=colors_s[cls], marker=markers_s[cls],
                   s=40, alpha=0.5, label=f"Synth {class_names[cls]}", edgecolors="white", linewidths=0.3)

    ax.set_title("t-SNE: Real vs Synthetic\n(good = classes overlap, not separated)",
                 color="white", fontsize=11)
    ax.set_facecolor("#161B22")
    ax.tick_params(colors="#AAAAAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.legend(fontsize=7, facecolor="#1C2128", labelcolor="white",
              ncol=2, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print(f"  t-SNE plot saved → {save_path}")


# ============================================================
# 6. TSTR — Train-Synthetic Test-Real
# ============================================================
def tstr_evaluation(X_real, y_real, X_syn, y_syn, X_test, y_test):
    """
    Train a simple classifier on synthetic data only, test on real data.
    Compare against training on real data (TRTR).
    If TSTR accuracy ≈ TRTR accuracy → synthetic data is useful.
    """
    print("\n" + "="*55)
    print("  6. TSTR — Train Synthetic, Test Real")
    print("="*55)

    def flatten(X):
        return X.reshape(len(X), -1)

    scaler = StandardScaler()

    # TRTR (baseline)
    X_tr_f  = scaler.fit_transform(flatten(X_real))
    X_tst_f = scaler.transform(flatten(X_test))
    clf_real = LogisticRegression(max_iter=1000, C=0.1)
    clf_real.fit(X_tr_f, y_real)
    trtr_acc = accuracy_score(y_test, clf_real.predict(X_tst_f))

    # TSTR
    X_syn_f = scaler.fit_transform(flatten(X_syn))
    X_tst_f = scaler.transform(flatten(X_test))
    clf_syn  = LogisticRegression(max_iter=1000, C=0.1)
    clf_syn.fit(X_syn_f, y_syn)
    tstr_acc = accuracy_score(y_test, clf_syn.predict(X_tst_f))

    ratio = tstr_acc / trtr_acc if trtr_acc > 0 else 0

    print(f"\n  TRTR (real train  → real test): {trtr_acc*100:.1f}%")
    print(f"  TSTR (synth train → real test): {tstr_acc*100:.1f}%")
    print(f"  TSTR/TRTR ratio               : {ratio:.2f}")

    if ratio >= 0.85:
        verdict = "✔ EXCELLENT — synthetic data is highly useful"
    elif ratio >= 0.70:
        verdict = "✔ GOOD      — synthetic data captures most structure"
    elif ratio >= 0.50:
        verdict = "⚠  FAIR      — some structure captured, train longer"
    else:
        verdict = "✗  POOR      — synthetic data not yet useful, train more"

    print(f"\n  Verdict: {verdict}")
    return tstr_acc, trtr_acc


# ============================================================
# SUMMARY REPORT
# ============================================================
def print_summary(rmse, fd, tstr_acc, trtr_acc):
    print("\n" + "="*55)
    print("  QUALITY SUMMARY")
    print("="*55)
    print(f"  RMSE              : {rmse:.4f}   (target < 0.25)")
    print(f"  Fréchet Distance  : {fd:.4f}   (target < 1.0)")
    print(f"  TSTR accuracy     : {tstr_acc*100:.1f}%")
    print(f"  TRTR accuracy     : {trtr_acc*100:.1f}%")
    print(f"  TSTR/TRTR ratio   : {tstr_acc/trtr_acc:.2f}   (target > 0.85)")

    passed = sum([rmse < 0.25, fd < 1.0, tstr_acc / trtr_acc >= 0.85])
    print(f"\n  Passed {passed}/3 quality checks")

    if passed == 3:
        print("  ✔ Synthesis is GOOD — safe to use for training")
    elif passed == 2:
        print("  ⚠ Synthesis is DECENT — consider more epochs")
    else:
        print("  ✗ Synthesis needs work — increase epochs or check architecture")
    print("="*55)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    PLOT_DIR = Path("../reports").resolve()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    X_real, y_real, X_syn, y_syn, X_test, y_test = load_all()

    check_statistics(X_real, y_real, X_syn, y_syn)
    rmse            = compute_rmse(X_real, y_real, X_syn, y_syn)
    fd              = compute_fd  (X_real, y_real, X_syn, y_syn)

    plot_psd  (X_real, y_real, X_syn, y_syn, PLOT_DIR / "psd_comparison.png")
    plot_tsne (X_real, y_real, X_syn, y_syn, PLOT_DIR / "tsne_comparison.png")

    tstr_acc, trtr_acc = tstr_evaluation(X_real, y_real, X_syn, y_syn, X_test, y_test)

    print_summary(rmse, fd, tstr_acc, trtr_acc)

    print(f"\nPlots saved to: {PLOT_DIR}")