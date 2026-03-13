"""
Part 1: XOR Problem — Neural Network from Scratch
==================================================
Proves that a simple 2-layer NN can solve XOR,
which is impossible for any linear classifier.

Architecture: 2 → 4 → 1  (input → hidden → output)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  Activation Functions
# ─────────────────────────────────────────────

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z):
    return np.tanh(z)

def tanh_deriv(z):
    return 1 - np.tanh(z)**2

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)


# ─────────────────────────────────────────────
#  Neural Network Class
# ─────────────────────────────────────────────

class NeuralNetworkXOR:
    """
    Fully-connected neural network: 2 → hidden_size → 1
    Trained with mini-batch gradient descent + backprop.
    """

    def __init__(self, hidden_size=4, activation='tanh', lr=0.1, seed=42):
        np.random.seed(seed)
        self.hidden_size = hidden_size
        self.lr = lr
        self.activation_name = activation

        # Xavier initialisation
        self.W1 = np.random.randn(2, hidden_size) * np.sqrt(2 / 2)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, 1))

        # Choose activation
        if activation == 'tanh':
            self._act, self._act_d = tanh, tanh_deriv
        elif activation == 'relu':
            self._act, self._act_d = relu, relu_deriv
        else:
            self._act, self._act_d = sigmoid, sigmoid_deriv

        # History
        self.losses = []
        self.accuracies = []
        self.weight_history = []   # snapshot W1 every N steps

    # ── Forward Pass ──
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self._act(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    # ── Compute Loss (BCE) ──
    def loss(self, y_pred, y_true):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # ── Backward Pass ──
    def backward(self, X, y_true):
        m = X.shape[0]
        dA2 = (self.A2 - y_true) / m          # dL/dA2
        dZ2 = dA2 * sigmoid_deriv(self.Z2)
        dW2 = self.A1.T @ dZ2
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self._act_d(self.Z1)
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # ── Training Loop ──
    def train(self, X, y, epochs=10000, verbose=True, snapshot_every=500):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            l = self.loss(y_pred, y)
            self.losses.append(l)
            acc = np.mean((y_pred > 0.5).astype(int) == y)
            self.accuracies.append(acc)
            self.backward(X, y)

            if epoch % snapshot_every == 0:
                self.weight_history.append(self.W1.copy())

            if verbose and epoch % 1000 == 0:
                print(f"  Epoch {epoch:>6}  |  Loss: {l:.6f}  |  Acc: {acc*100:.1f}%")

        return self

    # ── Predict ──
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# ─────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────

DARK_BG  = '#0d0d0d'
ACCENT1  = '#00f5c8'   # cyan-green
ACCENT2  = '#ff3f7a'   # pink-red
ACCENT3  = '#ffd93d'   # yellow
ACCENT4  = '#6a5acd'   # purple
GRID_COL = '#1e1e2e'

plt.rcParams.update({
    'figure.facecolor': DARK_BG,
    'axes.facecolor':   DARK_BG,
    'axes.edgecolor':   '#333355',
    'axes.labelcolor':  '#cccccc',
    'xtick.color':      '#777799',
    'ytick.color':      '#777799',
    'text.color':       '#eeeeee',
    'grid.color':       GRID_COL,
    'grid.alpha':       0.4,
    'font.family':      'monospace',
})


def plot_all(model, X, y, save_path='xor_results.png'):
    """Single figure with 4 panels."""
    fig = plt.figure(figsize=(18, 12), facecolor=DARK_BG)
    fig.suptitle('XOR — Neural Network from Scratch', fontsize=20,
                 color=ACCENT1, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # ── 1. Decision Boundary ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 400),
                         np.linspace(-0.3, 1.3, 400))
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    zz = model.forward(grid_pts).reshape(xx.shape)

    cmap_boundary = LinearSegmentedColormap.from_list(
        'xor', [ACCENT2, DARK_BG, ACCENT1], N=256)
    ax1.contourf(xx, yy, zz, levels=50, cmap=cmap_boundary, alpha=0.9)
    ax1.contour(xx, yy, zz, levels=[0.5], colors=[ACCENT3], linewidths=2)

    colors = [ACCENT1 if yi == 1 else ACCENT2 for yi in y.flatten()]
    ax1.scatter(X[:, 0], X[:, 1], c=colors, s=200, zorder=5,
                edgecolors='white', linewidths=1.5)

    for xi, yi_val, yi_label in zip(X, y.flatten(), ['0', '1', '1', '0']):
        ax1.annotate(f'({int(xi[0])},{int(xi[1])})→{yi_label}',
                     xi + 0.04, fontsize=8, color='white', alpha=0.8)

    ax1.set_title('Decision Boundary', color=ACCENT1, fontsize=12, pad=8)
    ax1.set_xlabel('x₁');  ax1.set_ylabel('x₂')
    ax1.grid(True, alpha=0.2)

    # ── 2. Loss Curve ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    epochs_arr = np.arange(len(model.losses))
    ax2.plot(epochs_arr, model.losses, color=ACCENT2, linewidth=1.5, label='BCE Loss')
    ax2.fill_between(epochs_arr, model.losses, alpha=0.15, color=ACCENT2)
    ax2.set_yscale('log')
    ax2.set_title('Training Loss (log scale)', color=ACCENT1, fontsize=12, pad=8)
    ax2.set_xlabel('Epoch');  ax2.set_ylabel('Loss')
    ax2.legend(facecolor='#1e1e2e', edgecolor='#444')
    ax2.grid(True, alpha=0.3)

    # ── 3. Accuracy Curve ─────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(epochs_arr, np.array(model.accuracies)*100,
             color=ACCENT1, linewidth=1.5, label='Accuracy %')
    ax3.fill_between(epochs_arr, np.array(model.accuracies)*100,
                     alpha=0.15, color=ACCENT1)
    ax3.axhline(100, color=ACCENT3, linestyle='--', linewidth=1, alpha=0.7, label='100%')
    ax3.set_title('Training Accuracy', color=ACCENT1, fontsize=12, pad=8)
    ax3.set_xlabel('Epoch');  ax3.set_ylabel('Accuracy (%)')
    ax3.set_ylim(0, 105)
    ax3.legend(facecolor='#1e1e2e', edgecolor='#444')
    ax3.grid(True, alpha=0.3)

    # ── 4. Weight Evolution (W1) ──────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    snapshots = np.array(model.weight_history)   # (T, 2, H)
    T = snapshots.shape[0]
    colors_w = [ACCENT1, ACCENT2, ACCENT3, ACCENT4,
                '#ff8c00', '#00bfff', '#7fff00', '#ff69b4']
    for h in range(model.hidden_size):
        for inp in range(2):
            ax4.plot(snapshots[:, inp, h],
                     color=colors_w[h % len(colors_w)],
                     alpha=0.7, linewidth=1.2,
                     label=f'W1[{inp},{h}]' if inp == 0 else '')
    ax4.set_title('W1 Weight Evolution', color=ACCENT1, fontsize=12, pad=8)
    ax4.set_xlabel('Snapshot');  ax4.set_ylabel('Weight Value')
    ax4.legend(fontsize=7, facecolor='#1e1e2e', edgecolor='#444', ncol=2)
    ax4.grid(True, alpha=0.3)

    # ── 5. Network Architecture Diagram ───────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_xlim(0, 10);  ax5.set_ylim(0, 10)
    ax5.axis('off')
    ax5.set_title('Network Architecture', color=ACCENT1, fontsize=12, pad=8)

    layer_x = [2, 5, 8]
    layer_nodes = [2, model.hidden_size, 1]
    node_pos = {}
    for li, (lx, n) in enumerate(zip(layer_x, layer_nodes)):
        ys = np.linspace(2, 8, n)
        for ni, ny in enumerate(ys):
            node_pos[(li, ni)] = (lx, ny)
            col = [ACCENT1, ACCENT3, ACCENT2][li]
            circ = plt.Circle((lx, ny), 0.38, color=col, zorder=4, alpha=0.9)
            ax5.add_patch(circ)
            ax5.text(lx, ny, str(ni+1), ha='center', va='center',
                     fontsize=8, color='black', fontweight='bold', zorder=5)

    for li in range(len(layer_nodes)-1):
        for ni in range(layer_nodes[li]):
            for nj in range(layer_nodes[li+1]):
                x0, y0 = node_pos[(li, ni)]
                x1, y1 = node_pos[(li+1, nj)]
                ax5.plot([x0, x1], [y0, y1], color='#444466',
                         linewidth=0.8, zorder=2, alpha=0.6)

    for lx, label in zip(layer_x, ['Input\n(2)', f'Hidden\n({model.hidden_size})', 'Output\n(1)']):
        ax5.text(lx, 1.2, label, ha='center', fontsize=8,
                 color='#aaaacc', va='top')

    # ── 6. Predictions Table ──────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    ax6.set_title('Final Predictions', color=ACCENT1, fontsize=12, pad=8)

    preds = model.forward(X)
    table_data = []
    for xi, yi, pi in zip(X, y.flatten(), preds.flatten()):
        pred_label = 1 if pi > 0.5 else 0
        ok = '✓' if pred_label == yi else '✗'
        table_data.append([f'({int(xi[0])}, {int(xi[1])})',
                            str(int(yi)), f'{pi:.4f}', str(pred_label), ok])

    cols = ['Input', 'True', 'P(1)', 'Pred', 'OK?']
    tbl = ax6.table(cellText=table_data, colLabels=cols,
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.4, 2.2)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor('#1a1a2e' if row > 0 else '#2a2a4e')
        cell.set_edgecolor('#444466')
        cell.set_text_props(color='white')
        if row > 0 and col == 4:
            txt = table_data[row-1][4]
            cell.set_facecolor('#0d3320' if txt == '✓' else '#3d0d1a')

    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    print(f"  ✓ Saved → {save_path}")
    plt.close()


def plot_linear_comparison(X, y, save_path='xor_linear_fail.png'):
    """Show why linear classifiers fail on XOR."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=DARK_BG)
    fig.suptitle('Why Linear Classifiers Fail on XOR', fontsize=15,
                 color=ACCENT2, fontweight='bold')

    for ax, title in zip(axes, ['Linear Attempt (fails)', 'Neural Net (succeeds)']):
        ax.set_facecolor(DARK_BG)
        ax.set_xlim(-0.5, 1.5);  ax.set_ylim(-0.5, 1.5)
        colors_pts = [ACCENT1 if yi == 1 else ACCENT2 for yi in y.flatten()]
        ax.scatter(X[:, 0], X[:, 1], c=colors_pts, s=300,
                   edgecolors='white', linewidths=2, zorder=4)
        ax.set_xlabel('x₁');  ax.set_ylabel('x₂')
        ax.set_title(title, color=ACCENT3, fontsize=11, pad=8)
        ax.grid(True, alpha=0.2)
        for xi, yi_val in zip(X, ['(0,0)→0', '(0,1)→1', '(1,0)→1', '(1,1)→0']):
            ax.annotate(yi_val, xi + np.array([0.05, 0.05]), fontsize=8, color='white')

    # Linear attempt — any line fails
    x_line = np.linspace(-0.5, 1.5, 100)
    for slope, intercept, alpha in [(1, 0, 0.4), (-1, 1, 0.4), (0, 0.5, 0.4)]:
        axes[0].plot(x_line, slope*x_line + intercept,
                     '--', color=ACCENT2, alpha=alpha, linewidth=1)
    axes[0].text(0.5, -0.4, 'No single line can separate classes!',
                 ha='center', color=ACCENT2, fontsize=9, style='italic')

    # NN boundary
    nn = NeuralNetworkXOR(hidden_size=4, activation='tanh', lr=0.5)
    nn.train(X, y, epochs=20000, verbose=False)
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 300),
                         np.linspace(-0.5, 1.5, 300))
    zz = nn.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    cmap2 = LinearSegmentedColormap.from_list('xor2', [ACCENT2, DARK_BG, ACCENT1])
    axes[1].contourf(xx, yy, zz, levels=40, cmap=cmap2, alpha=0.85)
    axes[1].contour(xx, yy, zz, levels=[0.5], colors=[ACCENT3], linewidths=2)
    axes[1].scatter(X[:, 0], X[:, 1], c=colors_pts, s=300,
                    edgecolors='white', linewidths=2, zorder=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=DARK_BG, edgecolor='none')
    print(f"  ✓ Saved → {save_path}")
    plt.close()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 55)
    print("  Part 1 — XOR Neural Network from Scratch")
    print("=" * 55)

    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    print("\n[1/3] Training Neural Network (2→4→1, tanh)...")
    model = NeuralNetworkXOR(hidden_size=4, activation='tanh', lr=0.5, seed=42)
    model.train(X, y, epochs=10000, verbose=True, snapshot_every=200)

    print("\n[2/3] Final Results:")
    preds = model.predict(X)
    for xi, yi, pi in zip(X, y.flatten(), model.forward(X).flatten()):
        print(f"  {xi} → True: {int(yi)} | P(1): {pi:.4f} | Pred: {int(pi>0.5)}")

    final_loss = model.losses[-1]
    final_acc = model.accuracies[-1]
    print(f"\n  Final Loss: {final_loss:.6f}")
    print(f"  Final Accuracy: {final_acc*100:.1f}%")

    print("\n[3/3] Generating Visualisations...")
    plot_all(model, X, y, save_path='part1_xor/xor_results.png')
    plot_linear_comparison(X, y, save_path='part1_xor/xor_linear_fail.png')

    print("\n✓ Part 1 complete!")
