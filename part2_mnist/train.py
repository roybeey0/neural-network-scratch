"""
Part 2: MNIST Train & Visualise
================================
Downloads MNIST via OpenML (sklearn-free fallback uses urllib + gzip),
trains the NN, and produces 6 publication-quality dark-theme charts.

Run:  python part2_mnist/train.py
"""

import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from neural_network import MNISTNeuralNet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# ─── dark theme ───────────────────────────────
DARK_BG  = '#0d0d0d'
CARD_BG  = '#111122'
ACCENT1  = '#00f5c8'
ACCENT2  = '#ff3f7a'
ACCENT3  = '#ffd93d'
ACCENT4  = '#6a5acd'
ACCENT5  = '#ff8c42'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   CARD_BG,
    'axes.edgecolor':   '#2a2a4a',
    'axes.labelcolor':  '#cccccc',
    'xtick.color':      '#777799',
    'ytick.color':      '#777799',
    'text.color':       '#eeeeee',
    'grid.color':       '#1e1e2e',
    'grid.alpha':       0.4,
    'font.family':      'monospace',
})

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output_charts')
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────

def load_mnist():
    """Load MNIST. Try local cache first, then download via urllib."""
    cache_path = os.path.join(os.path.dirname(__file__), 'mnist_cache.npz')

    if os.path.exists(cache_path):
        print("  Loading MNIST from local cache...")
        data = np.load(cache_path)
        return data['X_train'], data['y_train'], data['X_test'], data['y_test']

    print("  Downloading MNIST via urllib (one-time ~11 MB)...")
    import urllib.request
    import gzip
    import struct

    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images':  't10k-images-idx3-ubyte.gz',
        'test_labels':  't10k-labels-idx1-ubyte.gz',
    }

    raw = {}
    for key, fname in files.items():
        url = base_url + fname
        gz_path = os.path.join(os.path.dirname(__file__), fname)
        print(f"    ↓ {fname}")
        try:
            urllib.request.urlretrieve(url, gz_path)
        except Exception:
            # Fallback: try OpenML CSV
            return load_mnist_openml(cache_path)
        with gzip.open(gz_path, 'rb') as f:
            raw[key] = f.read()
        os.remove(gz_path)

    def parse_images(raw_bytes):
        offset = 16
        n = struct.unpack('>I', raw_bytes[4:8])[0]
        return np.frombuffer(raw_bytes[offset:], dtype=np.uint8).reshape(n, 784)

    def parse_labels(raw_bytes):
        return np.frombuffer(raw_bytes[8:], dtype=np.uint8)

    X_train = parse_images(raw['train_images']).astype(float) / 255.0
    y_train = parse_labels(raw['train_labels'])
    X_test  = parse_images(raw['test_images']).astype(float)  / 255.0
    y_test  = parse_labels(raw['test_labels'])

    np.savez_compressed(cache_path,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test,  y_test=y_test)
    return X_train, y_train, X_test, y_test


def load_mnist_openml(cache_path):
    """Fallback: load via fetch_openml (needs scikit-learn data only, no model)."""
    print("  Using OpenML fallback...")
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(float) / 255.0
        y = mnist.target.astype(int)
        X_train, y_train = X[:60000], y[:60000]
        X_test,  y_test  = X[60000:], y[60000:]
        np.savez_compressed(cache_path,
                            X_train=X_train, y_train=y_train,
                            X_test=X_test, y_test=y_test)
        return X_train, y_train, X_test, y_test
    except Exception as e:
        # Last resort: tiny synthetic MNIST-like data for demo
        print(f"  WARNING: Could not download real MNIST ({e}).")
        print("  Generating synthetic data for structure demo...")
        return generate_synthetic_mnist()


def generate_synthetic_mnist():
    """Generate structured synthetic data that mimics MNIST dimensions."""
    np.random.seed(42)
    n_train, n_test = 10000, 2000
    X_train = np.random.randn(n_train, 784) * 0.3 + 0.5
    X_train = np.clip(X_train, 0, 1)
    y_train = np.random.randint(0, 10, n_train)
    X_test  = np.random.randn(n_test, 784) * 0.3 + 0.5
    X_test  = np.clip(X_test, 0, 1)
    y_test  = np.random.randint(0, 10, n_test)
    print("  [SYNTHETIC DATA — for real results use actual MNIST]")
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────

def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    plt.close(fig)
    print(f"  ✓ {name}")
    return path


# Chart 1 ─ Training Curves
def plot_training_curves(nn):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG)
    fig.suptitle('MNIST — Training Curves', fontsize=16,
                 color=ACCENT1, fontweight='bold', y=1.02)

    epochs = np.arange(1, len(nn.train_losses) + 1)

    # Loss
    ax1.plot(epochs, nn.train_losses, color=ACCENT2, linewidth=2, label='Train Loss')
    ax1.plot(epochs, nn.val_losses,   color=ACCENT1, linewidth=2, label='Val Loss', linestyle='--')
    ax1.fill_between(epochs, nn.train_losses, alpha=0.1, color=ACCENT2)
    ax1.fill_between(epochs, nn.val_losses,   alpha=0.1, color=ACCENT1)
    ax1.set_title('Cross-Entropy Loss', color=ACCENT3, fontsize=12)
    ax1.set_xlabel('Epoch');  ax1.set_ylabel('Loss')
    ax1.legend(facecolor='#1a1a2e', edgecolor='#444')
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, np.array(nn.train_accs)*100,
             color=ACCENT2, linewidth=2, label='Train Acc')
    ax2.plot(epochs, np.array(nn.val_accs)*100,
             color=ACCENT1, linewidth=2, label='Val Acc', linestyle='--')
    ax2.fill_between(epochs, np.array(nn.train_accs)*100, alpha=0.1, color=ACCENT2)
    ax2.fill_between(epochs, np.array(nn.val_accs)*100,   alpha=0.1, color=ACCENT1)
    best_val = max(nn.val_accs) * 100
    best_ep  = np.argmax(nn.val_accs) + 1
    ax2.axhline(best_val, color=ACCENT3, linestyle=':', linewidth=1.5,
                label=f'Best Val {best_val:.2f}% (ep{best_ep})')
    ax2.set_title('Accuracy', color=ACCENT3, fontsize=12)
    ax2.set_xlabel('Epoch');  ax2.set_ylabel('Accuracy (%)')
    ax2.legend(facecolor='#1a1a2e', edgecolor='#444')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, '01_training_curves.png')


# Chart 2 ─ Confusion Matrix
def plot_confusion_matrix(nn, X_test, y_test):
    preds = nn.predict(X_test)
    cm = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_test, preds):
        cm[true][pred] += 1

    fig, ax = plt.subplots(figsize=(10, 9), facecolor=DARK_BG)
    fig.suptitle('Confusion Matrix — Test Set', fontsize=16,
                 color=ACCENT1, fontweight='bold')

    cmap = LinearSegmentedColormap.from_list('cm', [DARK_BG, '#1a1a4e', ACCENT4, ACCENT1])
    im = ax.imshow(cm, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(10):
        for j in range(10):
            val = cm[i, j]
            color = 'white' if val < cm.max() * 0.6 else 'black'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold' if i == j else 'normal')

    ax.set_xticks(range(10));  ax.set_yticks(range(10))
    ax.set_xticklabels(range(10));  ax.set_yticklabels(range(10))
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # Per-class accuracy
    per_class = [cm[i, i] / cm[i].sum() * 100 for i in range(10)]
    overall   = np.mean(preds == y_test) * 100
    ax.set_title(f'Overall Accuracy: {overall:.2f}%', color=ACCENT3, fontsize=12, pad=8)

    plt.tight_layout()
    save_fig(fig, '02_confusion_matrix.png')
    return cm


# Chart 3 ─ Sample Predictions
def plot_sample_predictions(nn, X_test, y_test, n=40):
    preds  = nn.predict(X_test)
    probas = nn.predict_proba(X_test)

    rows, cols = 5, 8
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10), facecolor=DARK_BG)
    fig.suptitle('Sample Predictions (green=correct, red=wrong)',
                 fontsize=14, color=ACCENT1, fontweight='bold')

    indices = np.random.choice(len(X_test), rows * cols, replace=False)

    for ax, idx in zip(axes.flat, indices):
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='inferno', interpolation='nearest')
        pred  = preds[idx]
        true  = y_test[idx]
        conf  = probas[idx, pred] * 100
        color = ACCENT1 if pred == true else ACCENT2
        ax.set_title(f'P:{pred} T:{true}\n{conf:.0f}%',
                     fontsize=7, color=color, pad=2)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.tight_layout()
    save_fig(fig, '03_sample_predictions.png')


# Chart 4 ─ Misclassified Examples
def plot_misclassified(nn, X_test, y_test, n=32):
    preds  = nn.predict(X_test)
    probas = nn.predict_proba(X_test)
    wrong  = np.where(preds != y_test)[0]

    rows, cols = 4, 8
    n = min(n, len(wrong))
    indices = wrong[:n]

    fig, axes = plt.subplots(rows, cols, figsize=(16, 8), facecolor=DARK_BG)
    fig.suptitle(f'Misclassified Examples  ({len(wrong)}/{len(y_test)} wrong, '
                 f'{len(wrong)/len(y_test)*100:.2f}% error rate)',
                 fontsize=13, color=ACCENT2, fontweight='bold')

    for ax, idx in zip(axes.flat, indices):
        img = X_test[idx].reshape(28, 28)
        ax.imshow(img, cmap='magma', interpolation='nearest')
        pred = preds[idx]
        true = y_test[idx]
        conf = probas[idx, pred] * 100
        ax.set_title(f'✗ P:{pred}  T:{true}\n{conf:.0f}%',
                     fontsize=7, color=ACCENT3, pad=2)
        ax.axis('off')

    # Fill blank panels
    for ax in list(axes.flat)[n:]:
        ax.axis('off')

    plt.tight_layout()
    save_fig(fig, '04_misclassified.png')


# Chart 5 ─ Weight Heatmaps (first layer)
def plot_weight_heatmaps(nn, n=64):
    """Visualise first-layer weights as 28×28 digit detectors."""
    W1 = nn.W[0]   # (784, 256)
    n  = min(n, W1.shape[1])

    rows = cols = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 14), facecolor=DARK_BG)
    fig.suptitle('First-Layer Weight Heatmaps (Learned Filters)',
                 fontsize=15, color=ACCENT1, fontweight='bold')

    cmap_w = LinearSegmentedColormap.from_list(
        'weights', [ACCENT2, DARK_BG, ACCENT1], N=256)

    for i, ax in enumerate(axes.flat):
        if i < n:
            w = W1[:, i].reshape(28, 28)
            vmax = np.abs(w).max()
            ax.imshow(w, cmap=cmap_w, vmin=-vmax, vmax=vmax,
                      interpolation='nearest')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout(pad=0.3)
    save_fig(fig, '05_weight_heatmaps.png')


# Chart 6 ─ Per-Class Accuracy Bar Chart
def plot_per_class_accuracy(nn, X_test, y_test):
    preds = nn.predict(X_test)
    per_class_acc = []
    per_class_n   = []
    for c in range(10):
        mask = y_test == c
        acc  = np.mean(preds[mask] == c) * 100
        per_class_acc.append(acc)
        per_class_n.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=DARK_BG)
    fig.suptitle('Per-Class Performance', fontsize=16,
                 color=ACCENT1, fontweight='bold')

    bar_colors = [ACCENT1 if a >= 97 else ACCENT3 if a >= 95 else ACCENT2
                  for a in per_class_acc]

    bars = ax1.bar(range(10), per_class_acc, color=bar_colors,
                   edgecolor='#333355', linewidth=0.8, width=0.7)
    ax1.axhline(np.mean(per_class_acc), color='white', linestyle='--',
                linewidth=1.5, alpha=0.6, label=f'Mean {np.mean(per_class_acc):.2f}%')
    ax1.set_ylim(90, 101)
    ax1.set_xticks(range(10))
    ax1.set_xlabel('Digit Class')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy per Digit', color=ACCENT3, fontsize=12)
    ax1.legend(facecolor='#1a1a2e', edgecolor='#444')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, per_class_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9, color='white')

    # Confidence distribution per class
    probas = nn.predict_proba(X_test)
    box_data = [probas[y_test == c, c] * 100 for c in range(10)]
    bp = ax2.boxplot(box_data, patch_artist=True,
                     medianprops=dict(color=ACCENT3, linewidth=2),
                     whiskerprops=dict(color='#666688'),
                     capprops=dict(color='#666688'),
                     flierprops=dict(marker='o', markersize=2,
                                     markerfacecolor=ACCENT2, alpha=0.4))

    for patch, col in zip(bp['boxes'], bar_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    ax2.set_xticks(range(1, 11))
    ax2.set_xticklabels(range(10))
    ax2.set_xlabel('Digit Class')
    ax2.set_ylabel('Confidence (%)')
    ax2.set_title('Confidence Distribution', color=ACCENT3, fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_fig(fig, '06_per_class_accuracy.png')


# Chart 7 ─ Summary Dashboard
def plot_summary(nn, X_test, y_test):
    preds = nn.predict(X_test)
    final_val_acc = nn.val_accs[-1] * 100
    best_val_acc  = max(nn.val_accs) * 100

    fig = plt.figure(figsize=(16, 9), facecolor=DARK_BG)
    fig.suptitle('MNIST Neural Network — Results Dashboard',
                 fontsize=18, color=ACCENT1, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Big metrics ──────────────────────────────────
    metrics = [
        ('Val Accuracy', f'{final_val_acc:.2f}%', ACCENT1),
        ('Best Val Acc', f'{best_val_acc:.2f}%',  ACCENT3),
        ('Parameters',  f'{sum(w.size for w in nn.W):,}', ACCENT5),
        ('Epochs',      str(len(nn.train_losses)), ACCENT4),
    ]
    for i, (label, val, col) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#0a0a1a')
        ax.axis('off')
        ax.text(0.5, 0.62, val,   ha='center', va='center', fontsize=24,
                color=col, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.28, label, ha='center', va='center', fontsize=10,
                color='#888899', transform=ax.transAxes)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(col)
            spine.set_linewidth(1.5)

    # ── Loss curve (small) ───────────────────────────
    ax_loss = fig.add_subplot(gs[1, 0:2])
    eps = np.arange(1, len(nn.train_losses)+1)
    ax_loss.plot(eps, nn.train_losses, ACCENT2, linewidth=1.5, label='Train')
    ax_loss.plot(eps, nn.val_losses,   ACCENT1, linewidth=1.5, label='Val', linestyle='--')
    ax_loss.set_title('Loss Curve', color=ACCENT3, fontsize=11)
    ax_loss.set_xlabel('Epoch');  ax_loss.set_ylabel('Loss')
    ax_loss.legend(facecolor='#1a1a2e', edgecolor='#444', fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    # ── Accuracy curve (small) ───────────────────────
    ax_acc = fig.add_subplot(gs[1, 2:4])
    ax_acc.plot(eps, np.array(nn.train_accs)*100, ACCENT2, linewidth=1.5, label='Train')
    ax_acc.plot(eps, np.array(nn.val_accs)*100,   ACCENT1, linewidth=1.5, label='Val', linestyle='--')
    ax_acc.set_title('Accuracy Curve', color=ACCENT3, fontsize=11)
    ax_acc.set_xlabel('Epoch');  ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.legend(facecolor='#1a1a2e', edgecolor='#444', fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig(fig, '00_summary_dashboard.png')


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  Part 2 — MNIST Neural Network from Scratch")
    print("=" * 60)

    # ── Load Data ──
    print("\n[1/4] Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # Use a validation split from training set (handle small synthetic data too)
    n_total = len(X_train)
    n_val   = min(10000, max(500, n_total // 6))
    X_val,   y_val   = X_train[n_total - n_val:], y_train[n_total - n_val:]
    X_train, y_train = X_train[:n_total - n_val], y_train[:n_total - n_val]
    print(f"  Using {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    # ── Build Model ──
    print("\n[2/4] Building model: 784 → 256 → 128 → 10")
    nn = MNISTNeuralNet(
        hidden_sizes=(256, 128),
        lr=0.08,
        momentum=0.9,
        dropout_rate=0.2,
        lr_decay=0.97,
        seed=42,
    )
    total_params = sum(w.size for w in nn.W)
    print(f"  Total parameters: {total_params:,}")

    # ── Train ──
    print("\n[3/4] Training (30 epochs)...")
    t0 = time.time()
    nn.train(X_train, y_train, X_val, y_val,
             epochs=30, batch_size=256, verbose=True)
    elapsed = time.time() - t0
    print(f"\n  ⏱  Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Evaluate ──
    test_preds = nn.predict(X_test)
    test_acc   = np.mean(test_preds == y_test) * 100
    best_val   = max(nn.val_accs) * 100
    print(f"\n  Test Accuracy : {test_acc:.2f}%")
    print(f"  Best Val Acc  : {best_val:.2f}%")

    # ── Save model ──
    nn.save(os.path.join(os.path.dirname(__file__), 'model.pkl'))

    # ── Visualisations ──
    print("\n[4/4] Generating visualisations...")
    plot_training_curves(nn)
    plot_confusion_matrix(nn, X_test, y_test)
    plot_sample_predictions(nn, X_test, y_test)
    plot_misclassified(nn, X_test, y_test)
    plot_weight_heatmaps(nn)
    plot_per_class_accuracy(nn, X_test, y_test)
    plot_summary(nn, X_test, y_test)

    print(f"\n✓ All charts saved → {OUT_DIR}/")
    print(f"✓ Final Test Accuracy: {test_acc:.2f}%")
