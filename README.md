# Self-Pruning Neural Network using Learnable Gates

## Overview

This project implements a neural network that learns to **prune itself during training** using learnable gating mechanisms.

Each connection is controlled by a gate, and sparsity is encouraged using L1 regularization.

---

## Key Idea

Each weight is modulated by a learnable gate:

```
g_ij = sigmoid(s_ij / T)
w̃_ij = w_ij × g_ij
```

The model learns which connections are important and removes redundant ones automatically.

---

## Loss Function

```
L = CE Loss + λ × Sparsity Loss
Sparsity Loss = mean(|g_ij|)
```

* CE Loss → classification objective
* Sparsity Loss → encourages pruning
* λ → controls sparsity level

---

## Results

| λ    | Accuracy (%) | Sparsity (%) |
| ---- | ------------ | ------------ |
| 0.01 | 53.70        | 37.71        |
| 0.1  | 53.59        | 40.58        |
| 0.5  | 53.61        | 48.39        |
| 1.0  | 53.95        | 54.76        |
| 2.0  | 53.09        | 62.02        |

---

## Tradeoff

* Increasing λ → higher sparsity
* Accuracy remains stable (~53–54%)
* Up to **62% pruning** with minimal loss

---

## Visualization

### Accuracy vs Sparsity

![Tradeoff](results/tradeoff.png)

### Gate Distribution

![Distribution](results/gate_distribution.png)

---

## Key Insight

The model is **overparameterized**, meaning many connections are redundant.

The gating mechanism successfully removes these without affecting performance.

---

## Hard Pruning Validation

After converting soft gates into hard masks:

* Sparsity remains high
* Accuracy remains stable

This confirms that learned gates correspond to **actual removable connections**.

---

## How to Run

1. Open `notebook.ipynb`
2. Run all cells in Google Colab

---

## Tech Stack

* Python
* PyTorch
* NumPy
* Matplotlib

---

## Why This Matters

This approach enables:

* Model compression
* Efficient deployment
* Automatic architecture optimization

---

## Author
* kunal singh

Your Name
