"""
Part 1 Demo — Interactive XOR Exploration
==========================================
Experiments: different architectures, activations, learning rates.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from xor import NeuralNetworkXOR, plot_all, DARK_BG, ACCENT1, ACCENT2, ACCENT3

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)


def experiment_architectures():
    """Compare different hidden-layer sizes."""
    configs = [
        {'hidden_size': 2, 'label': '2→2→1'},
        {'hidden_size': 4, 'label': '2→4→1 ★'},
        {'hidden_size': 8, 'label': '2→8→1'},
        {'hidden_size': 16, 'label': '2→16→1'},
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), facecolor=DARK_BG)
    fig.suptitle('Architecture Comparison — XOR', fontsize=16,
                 color=ACCENT1, fontweight='bold')

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('xor', [ACCENT2, DARK_BG, ACCENT1])
    colors_pts = [ACCENT1 if yi == 1 else ACCENT2 for yi in y.flatten()]

    for i, cfg in enumerate(configs):
        nn = NeuralNetworkXOR(hidden_size=cfg['hidden_size'],
                              activation='tanh', lr=0.5, seed=42)
        nn.train(X, y, epochs=10000, verbose=False, snapshot_every=500)

        ax_top = axes[0][i]
        ax_top.set_facecolor(DARK_BG)
        xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 250),
                              np.linspace(-0.3, 1.3, 250))
        zz = nn.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax_top.contourf(xx, yy, zz, levels=40, cmap=cmap, alpha=0.9)
        ax_top.contour(xx, yy, zz, levels=[0.5],
                       colors=[ACCENT3], linewidths=2)
        ax_top.scatter(X[:, 0], X[:, 1], c=colors_pts,
                       s=150, edgecolors='white', linewidths=1.5, zorder=4)
        acc = nn.accuracies[-1]
        title_col = ACCENT1 if acc == 1.0 else ACCENT2
        ax_top.set_title(f'{cfg["label"]}\nAcc: {acc*100:.0f}%',
                         color=title_col, fontsize=10)
        ax_top.axis('off')

        ax_bot = axes[1][i]
        ax_bot.set_facecolor(DARK_BG)
        ax_bot.plot(nn.losses, color=ACCENT2, linewidth=1.2)
        ax_bot.set_yscale('log')
        ax_bot.set_title(f'Loss — {cfg["label"]}', color='#aaa', fontsize=9)
        ax_bot.set_xlabel('Epoch', fontsize=8)
        ax_bot.grid(True, alpha=0.2)
        for spine in ax_bot.spines.values():
            spine.set_edgecolor('#333355')

    plt.tight_layout()
    plt.savefig('part1_xor/xor_architectures.png',
                dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    print("  ✓ Saved → part1_xor/xor_architectures.png")
    plt.close()


def experiment_activations():
    """Compare activation functions."""
    activations = ['sigmoid', 'tanh', 'relu']
    results = {}
    for act in activations:
        nn = NeuralNetworkXOR(hidden_size=4, activation=act, lr=0.5, seed=0)
        nn.train(X, y, epochs=10000, verbose=False, snapshot_every=500)
        results[act] = nn

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK_BG)
    fig.suptitle('Activation Function Comparison — XOR', fontsize=14,
                 color=ACCENT1, fontweight='bold')

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('xor', [ACCENT2, DARK_BG, ACCENT1])
    colors_pts = [ACCENT1 if yi == 1 else ACCENT2 for yi in y.flatten()]

    for ax, act in zip(axes, activations):
        nn = results[act]
        ax.set_facecolor(DARK_BG)
        xx, yy = np.meshgrid(np.linspace(-0.3, 1.3, 300),
                              np.linspace(-0.3, 1.3, 300))
        zz = nn.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=40, cmap=cmap, alpha=0.9)
        ax.contour(xx, yy, zz, levels=[0.5],
                   colors=[ACCENT3], linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=colors_pts,
                   s=200, edgecolors='white', linewidths=1.5, zorder=4)
        acc = nn.accuracies[-1]
        ax.set_title(f'{act.upper()}  |  Acc: {acc*100:.0f}%  |  Loss: {nn.losses[-1]:.4f}',
                     color=ACCENT1, fontsize=11)
        ax.set_xlabel('x₁');  ax.set_ylabel('x₂')
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    plt.tight_layout()
    plt.savefig('part1_xor/xor_activations.png',
                dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    print("  ✓ Saved → part1_xor/xor_activations.png")
    plt.close()


if __name__ == '__main__':
    print("=" * 55)
    print("  Part 1 Demo — Architecture & Activation Experiments")
    print("=" * 55)

    print("\n[1/2] Comparing architectures...")
    experiment_architectures()

    print("\n[2/2] Comparing activation functions...")
    experiment_activations()

    print("\n✓ Demo complete! Check part1_xor/ for charts.")
