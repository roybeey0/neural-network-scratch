"""
Part 2: MNIST Digit Classification — Neural Network from Scratch
================================================================
Pure NumPy implementation. No PyTorch / TensorFlow / scikit-learn.

Architecture: 784 → 256 → 128 → 10
Techniques:
  - He initialisation
  - ReLU hidden + Softmax output
  - Cross-entropy loss
  - Mini-batch SGD with momentum
  - Dropout regularisation
  - Learning-rate decay
"""

import numpy as np
import pickle
import os


# ─────────────────────────────────────────────
#  Activation & Loss
# ─────────────────────────────────────────────

def relu(Z):
    return np.maximum(0, Z)

def relu_back(dA, Z):
    return dA * (Z > 0)

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true_onehot):
    """Cross-entropy loss. y_pred: (m, 10), y_true_onehot: (m, 10)."""
    eps = 1e-12
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred + eps), axis=1))

def to_onehot(y, num_classes=10):
    m = len(y)
    oh = np.zeros((m, num_classes), dtype=float)
    oh[np.arange(m), y] = 1.0
    return oh


# ─────────────────────────────────────────────
#  Neural Network
# ─────────────────────────────────────────────

class MNISTNeuralNet:
    """
    3-layer fully connected NN: 784 → 256 → 128 → 10

    Hyper-parameters
    ----------------
    hidden_sizes : tuple  – e.g. (256, 128)
    lr           : float  – initial learning rate
    momentum     : float  – SGD momentum (0.9 works well)
    dropout_rate : float  – dropout probability (0 = disabled)
    lr_decay     : float  – multiply lr by this every epoch
    """

    def __init__(
        self,
        hidden_sizes=(256, 128),
        lr=0.1,
        momentum=0.9,
        dropout_rate=0.2,
        lr_decay=0.97,
        seed=42,
    ):
        np.random.seed(seed)
        self.hidden_sizes = hidden_sizes
        self.lr0 = lr
        self.lr = lr
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.lr_decay = lr_decay

        # Build layer sizes: 784 → h1 → h2 → 10
        sizes = [784] + list(hidden_sizes) + [10]
        self.num_layers = len(sizes) - 1

        # He initialisation for ReLU layers
        self.W, self.b = [], []
        self.vW, self.vb = [], []   # velocity for momentum
        for i in range(self.num_layers):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            # He init: std = sqrt(2 / fan_in)
            W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.W.append(W)
            self.b.append(b)
            self.vW.append(np.zeros_like(W))
            self.vb.append(np.zeros_like(b))

        # History
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    # ── Forward Pass (supports dropout mask) ──
    def forward(self, X, training=False):
        self._cache = {'A0': X}
        A = X
        for i in range(self.num_layers - 1):
            Z = A @ self.W[i] + self.b[i]
            A = relu(Z)
            # Dropout (inverted dropout)
            if training and self.dropout_rate > 0:
                mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
                A *= mask / (1.0 - self.dropout_rate)
                self._cache[f'mask{i}'] = mask
            self._cache[f'Z{i+1}'] = Z
            self._cache[f'A{i+1}'] = A

        # Final layer — softmax
        Z_last = A @ self.W[-1] + self.b[-1]
        A_last = softmax(Z_last)
        self._cache[f'Z{self.num_layers}'] = Z_last
        self._cache[f'A{self.num_layers}'] = A_last
        return A_last

    # ── Backward Pass ──
    def backward(self, X, y_onehot):
        m = X.shape[0]
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        # Output layer gradient (softmax + CE combined)
        dZ = self._cache[f'A{self.num_layers}'] - y_onehot  # (m, 10)
        A_prev = self._cache[f'A{self.num_layers - 1}']
        grads_W[-1] = (A_prev.T @ dZ) / m
        grads_b[-1] = np.mean(dZ, axis=0, keepdims=True)
        dA = dZ @ self.W[-1].T

        # Hidden layers (reverse)
        for i in range(self.num_layers - 2, -1, -1):
            Z = self._cache[f'Z{i+1}']
            dZ = relu_back(dA, Z)
            # Apply dropout mask if present
            if f'mask{i}' in self._cache:
                dZ *= self._cache[f'mask{i}'] / (1.0 - self.dropout_rate)
            A_prev = self._cache[f'A{i}']
            grads_W[i] = (A_prev.T @ dZ) / m
            grads_b[i] = np.mean(dZ, axis=0, keepdims=True)
            dA = dZ @ self.W[i].T

        # SGD with momentum update
        for i in range(self.num_layers):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * grads_W[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * grads_b[i]
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    # ── Mini-batch Training ──
    def train(self, X_train, y_train, X_val, y_val,
              epochs=30, batch_size=128, verbose=True):

        n = X_train.shape[0]
        y_train_oh = to_onehot(y_train)

        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = np.random.permutation(n)
            X_shuf, y_shuf_oh = X_train[idx], y_train_oh[idx]

            # Mini-batches
            for start in range(0, n, batch_size):
                Xb = X_shuf[start:start + batch_size]
                yb = y_shuf_oh[start:start + batch_size]
                self.forward(Xb, training=True)
                self.backward(Xb, yb)

            # Epoch metrics (eval mode — no dropout)
            pred_train = self.forward(X_train, training=False)
            pred_val   = self.forward(X_val,   training=False)

            t_loss = cross_entropy(pred_train, y_train_oh)
            v_loss = cross_entropy(pred_val,   to_onehot(y_val))
            t_acc  = np.mean(np.argmax(pred_train, axis=1) == y_train)
            v_acc  = np.mean(np.argmax(pred_val,   axis=1) == y_val)

            self.train_losses.append(t_loss)
            self.val_losses.append(v_loss)
            self.train_accs.append(t_acc)
            self.val_accs.append(v_acc)

            # LR decay
            self.lr = self.lr0 * (self.lr_decay ** epoch)

            if verbose:
                v_acc_safe = v_acc if not np.isnan(v_acc) else 0.0
                bar = '█' * int(v_acc_safe * 20) + '░' * (20 - int(v_acc_safe * 20))
                print(f"  Epoch {epoch:>3}/{epochs}  |  "
                      f"Train {t_acc*100:5.2f}%  Val {v_acc*100:5.2f}%  "
                      f"|  Loss {v_loss:.4f}  |  [{bar}]  lr={self.lr:.5f}")

        return self

    # ── Predict ──
    def predict(self, X):
        return np.argmax(self.forward(X, training=False), axis=1)

    def predict_proba(self, X):
        return self.forward(X, training=False)

    # ── Save / Load ──
    def save(self, path):
        data = {
            'W': self.W, 'b': self.b,
            'hidden_sizes': self.hidden_sizes,
            'train_losses': self.train_losses,
            'val_losses':   self.val_losses,
            'train_accs':   self.train_accs,
            'val_accs':     self.val_accs,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  ✓ Model saved → {path}")

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        nn = cls(hidden_sizes=data['hidden_sizes'])
        nn.W = data['W']
        nn.b = data['b']
        nn.train_losses = data['train_losses']
        nn.val_losses   = data['val_losses']
        nn.train_accs   = data['train_accs']
        nn.val_accs     = data['val_accs']
        return nn
