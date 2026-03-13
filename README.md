# 🧠 Neural Network from Scratch — XOR & MNIST

<p align="center">
  <img src="part2_mnist/output_charts/00_summary_dashboard.png" alt="Summary Dashboard" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy_Only-No_ML_Libs-013243?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/MNIST-98.43%25_Acc-00f5c8?style=for-the-badge">
  <img src="https://img.shields.io/badge/XOR-100%25_Acc-ffd93d?style=for-the-badge">
  <img src="https://img.shields.io/badge/Training-1.8_min_CPU-ff3f7a?style=for-the-badge">
  <img src="https://img.shields.io/badge/Portfolio-roybeey.com-6a5acd?style=for-the-badge">
</p>

> **Pure NumPy neural network** — from forward pass to backprop, implemented from first principles.  
> No PyTorch. No TensorFlow. No scikit-learn models. Just math and code.

---

## 📌 Overview

This project demonstrates a complete deep learning pipeline implemented entirely from scratch using only **NumPy** and **Matplotlib**. Two progressively complex problems are tackled:

| | Part 1 — XOR | Part 2 — MNIST |
|---|---|---|
| **Problem** | Non-linear binary classification | 10-class digit recognition |
| **Architecture** | 2 → 4 → 1 | 784 → 256 → 128 → 10 |
| **Activation** | tanh / ReLU / sigmoid | ReLU + Softmax |
| **Optimiser** | Vanilla SGD | SGD + Momentum + LR Decay |
| **Regularisation** | — | Dropout (p=0.2) |
| **Accuracy** | **100%** | **98.43%** |
| **Training time** | < 1 sec | 1.8 min (CPU) |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/roybeey0/neural-network-scratch
cd neural-network-scratch

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Part 1 — XOR
python part1_xor/xor.py

# 4. Run Part 1 — Experiments demo
python part1_xor/demo.py

# 5. Run Part 2 — MNIST (downloads data automatically)
python part2_mnist/train.py
```

---

## 🗂️ Project Structure

```
neural-network-scratch/
├── part1_xor/
│   ├── xor.py                    ← Core NN + all XOR visualisations
│   ├── demo.py                   ← Architecture & activation experiments
│   ├── xor_results.png           ← Decision boundary + training curves
│   ├── xor_linear_fail.png       ← Why linear classifiers fail on XOR
│   └── xor_architectures.png     ← Architecture comparison grid
│
├── part2_mnist/
│   ├── neural_network.py         ← MNISTNeuralNet class (pure NumPy)
│   ├── train.py                  ← Training script + 7 visualisations
│   ├── model.pkl                 ← Saved weights (after training)
│   └── output_charts/
│       ├── 00_summary_dashboard.png
│       ├── 01_training_curves.png
│       ├── 02_confusion_matrix.png
│       ├── 03_sample_predictions.png
│       ├── 04_misclassified.png
│       ├── 05_weight_heatmaps.png
│       └── 06_per_class_accuracy.png
│
├── requirements.txt
└── README.md
```

---

## 🔬 Part 1 — XOR Problem

### Why XOR Matters

The XOR problem is the canonical proof that **linear classifiers are fundamentally limited**. A single perceptron can only draw one straight line — but XOR requires separating two groups that are **diagonally opposite**, impossible with any linear boundary.

```
  x₁  x₂  │  XOR
 ──────────┼──────
   0   0   │   0   ←─ class A
   0   1   │   1   ←─ class B
   1   0   │   1   ←─ class B
   1   1   │   0   ←─ class A
```

A neural network with even **one hidden layer** (Universal Approximation Theorem) can learn any non-linear boundary — including this one.

### Architecture

```
Input Layer     Hidden Layer    Output Layer
  x₁ ──────┐
            ├──→ [h₁]
  x₂ ──────┼──→ [h₂] ──→ [y]   (sigmoid → binary)
            ├──→ [h₃]
            └──→ [h₄]

Activation: tanh | Loss: Binary Cross-Entropy | LR: 0.5
```

### Results

```
Input    True   P(1)    Pred
(0, 0) →  0   | 0.0023  | 0   ✓
(0, 1) →  1   | 0.9981  | 1   ✓
(1, 0) →  1   | 0.9979  | 1   ✓
(1, 1) →  0   | 0.0019  | 0   ✓

Final Loss: 0.005234  |  Accuracy: 100%
```

<p align="center">
  <img src="part1_xor/xor_results.png" alt="XOR Results" width="90%">
</p>

<p align="center">
  <img src="part1_xor/xor_linear_fail.png" alt="Linear vs NN" width="80%">
</p>

---

## 🔢 Part 2 — MNIST Digit Classification

### Implementation Details

Every component is implemented from scratch:

**Forward Pass**
```python
Z1 = X  @ W1 + b1      # Linear transform
A1 = relu(Z1)           # ReLU activation
Z2 = A1 @ W2 + b2
A2 = relu(Z2)
Z3 = A2 @ W3 + b3
A3 = softmax(Z3)        # Probability distribution over 10 classes
```

**Backpropagation**
```python
# Output layer — softmax + cross-entropy combined gradient
dZ3 = A3 - y_onehot                      # (m, 10)
dW3 = (A2.T @ dZ3) / m

# Hidden layers — chain rule through ReLU
dA2 = dZ3 @ W3.T
dZ2 = dA2 * (Z2 > 0)                     # ReLU derivative
dW2 = (A1.T @ dZ2) / m
# ... and so on
```

**SGD with Momentum**
```python
v_W = momentum * v_W - lr * grad_W
W  += v_W                                # Accelerates convergence
```

**Inverted Dropout**
```python
mask = (random(shape) > dropout_rate)
A   *= mask / (1 - dropout_rate)         # Scale to preserve expectation
```

**He Initialisation**
```python
W = random.randn(fan_in, fan_out) * sqrt(2.0 / fan_in)   # Optimal for ReLU
```

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| Architecture | 784→256→128→10 | Balance capacity vs speed |
| Activation | ReLU + Softmax | ReLU avoids vanishing gradient |
| Learning rate | 0.08 | Starting LR |
| LR decay | 0.97/epoch | Smooth convergence |
| Momentum | 0.9 | Standard Nesterov-like acceleration |
| Dropout | p=0.2 | Light regularisation |
| Batch size | 256 | Fast GPU-less training |
| Epochs | 30 | Converges well before 30 |
| Initialisation | He | Designed for ReLU networks |

### Results

```
Test Accuracy : 98.43%
Best Val Acc  : 98.38%
Parameters    : 234,752
Training time : 1.8 min (CPU, no GPU)
```

<p align="center">
  <img src="part2_mnist/output_charts/01_training_curves.png" width="90%">
</p>

<p align="center">
  <img src="part2_mnist/output_charts/02_confusion_matrix.png" width="60%">
</p>

<p align="center">
  <img src="part2_mnist/output_charts/03_sample_predictions.png" width="90%">
</p>

<p align="center">
  <img src="part2_mnist/output_charts/05_weight_heatmaps.png" width="80%">
</p>

---

## 🧮 Mathematical Foundation

### Binary Cross-Entropy (XOR)
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1-y^{(i)}) \log(1-\hat{y}^{(i)}) \right]$$

### Categorical Cross-Entropy (MNIST)
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=0}^{9} y_k^{(i)} \log(\hat{y}_k^{(i)})$$

### Backpropagation (Chain Rule)
$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{m} A^{[l-1]T} \cdot \delta^{[l]}$$

---

## 📊 Visualisations

| Chart | Description |
|---|---|
| `xor_results.png` | Decision boundary, loss, accuracy, weights, predictions |
| `xor_linear_fail.png` | Side-by-side: linear classifier fails vs NN succeeds |
| `xor_architectures.png` | 4 architecture sizes compared |
| `00_summary_dashboard.png` | High-level metrics + training overview |
| `01_training_curves.png` | Train vs val loss & accuracy per epoch |
| `02_confusion_matrix.png` | Full 10×10 confusion matrix with counts |
| `03_sample_predictions.png` | 40 test samples with confidence |
| `04_misclassified.png` | Hard cases the model got wrong |
| `05_weight_heatmaps.png` | 64 first-layer neurons as 28×28 filters |
| `06_per_class_accuracy.png` | Per-digit accuracy + confidence boxplots |

---

## 👤 Author

**Roybeey** — Data Science & AI, UBAYA Surabaya  
Portfolio: [roybeey.com](https://roybeey.com)

**Other Projects**
- Trading Bot (OKX API + strategy backtesting)
- Genetic Algorithm Optimizer
- CodeT5 Code Generator
- Crypto Market EDA
- Python Blockchain + Wallet Analytics
- On-chain Whale Tracker

---

## 📄 License

MIT License — feel free to study, fork, and build on this.