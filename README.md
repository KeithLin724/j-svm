# j-SVM: GPU-Accelerated Parallel SVM using JAX

> ⭐ **If you find this project helpful, please give it a star on GitHub!**
>
> 📖 **[Project Introduction Slides](https://gamma.app/docs/j-SVM-slide-s253x90o2ruq768)**

**j-SVM** is a dual-implementation Support Vector Machine library that provides both a pure **NumPy** backend (`SVM/`) for CPU execution and a **JAX**-accelerated backend (`JSVM/`) for GPU/TPU acceleration. The JAX backend leverages JIT compilation, automatic differentiation, and Nyström kernel approximation to scale SVMs to large datasets efficiently.

---

## 🏷️ Badges

### License & Runtime

![Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-^3.12-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![JAX](https://img.shields.io/badge/JAX-^0.6.0-8A2BE2?style=for-the-badge&logo=jax&logoColor=white)

### Platform

![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green)
![AMD_HIP](https://img.shields.io/badge/HIP-%23000000.svg?style=for-the-badge&logo=amd&logoColor=white&logoSize=auto)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

### Tools

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Poetry](https://img.shields.io/badge/Poetry-%233B82F6.svg?style=for-the-badge&logo=poetry&logoColor=0B3D8D)

---

## ✨ Features

- **Dual implementation** — Choose between NumPy (CPU) or JAX (GPU/TPU) backends with identical APIs
- **Multiple kernel support** — Linear, RBF (via Nyström approximation), and Polynomial kernels
- **Nyström kernel approximation** — Efficient low-rank approximation of large kernel matrices for scalable training
- **JIT-compiled forward pass** — JAX backend uses `jax.jit` for accelerated computation
- **Model persistence** — Save and load trained models via `joblib`
- **Clean OOP design** — Strategy pattern for pluggable kernel functions
- **Multi-class classification** — One-vs-One strategy via `MultiSupportVectorMachine`
- **Built-in dataset loaders** — Iris, Pima Indians Diabetes, Framingham Heart Study

---

## 🏗️ Architecture

```
j-svm/
├── core/                          # Base parameter definitions
│   ├── __init__.py
│   └── SVMParameter.py            # SVM hyperparameter container
│
├── JSVM/                          # JAX-accelerated implementation (GPU/TPU)
│   ├── __init__.py
│   ├── SupportVectorMachine.py    # Binary SVM — JAX backend
│   └── MultiSupportVectorMachine.py  # Multi-class (OvO) — JAX backend
│
├── SVM/                           # NumPy implementation (CPU)
│   ├── __init__.py
│   ├── SupportVectorMachine.py    # Binary SVM — NumPy backend
│   └── MultiSupportVectorMachine.py  # Multi-class (OvO) — NumPy backend
│
├── data/                          # Dataset loading utilities
│   ├── __init__.py
│   ├── DataUnit.py                # Unified dataset container
│   ├── IrisDataset.py             # Iris dataset loader
│   └── functional.py              # Data preprocessing helpers
│
├── utils/                         # Testing utilities
│   ├── __init__.py
│   └── test_utils.py
│
├── example/                       # Usage examples
│   ├── normal_svm.py              # NumPy SVM on iris
│   ├── normal_svm_jax.py          # JAX SVM on iris
│   ├── normal_multi_svm.py        # NumPy multi-class SVM
│   ├── normal_multi_svm_jax.py    # JAX multi-class SVM
│   ├── sample_jax_code.py         # JAX primitive demos
│   ├── sample_jax_code_2.py       # Advanced JAX demos
│   └── data_set_all_plus.py       # All-dataset benchmark
│
├── main.py                        # Entry point
├── run_JSVM.py                    # JAX runtime script
├── run_SVM.py                     # NumPy runtime script
└── pyproject.toml                 # Poetry project configuration
```

---

## 🚀 Quick Start

### Installation

```sh
git clone https://github.com/KeithLin724/j-svm.git
cd j-svm
pip install poetry
poetry install
poetry run python -c "from JSVM import SupportVectorMachine; SupportVectorMachine.warm_up()"
```

### Minimal Working Example (Iris Dataset)

```python
from JSVM import SupportVectorMachine   # or: from SVM import SupportVectorMachine
from data import IrisDataset

# Warm up JAX (only needed for JAX backend)
SupportVectorMachine.warm_up()

# Load data
data = IrisDataset()

# Train
model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})
model.train(data.train_x, data.train_y)

# Evaluate
print(f"Accuracy: {model.acc(data.test_x, data.test_y):.2%}")
```

Expected output (JAX backend, RBF kernel): ~96% accuracy on Iris test set.

---

## 📖 Usage

### J-SVM (JAX / GPU Backend)

```python
from JSVM import SupportVectorMachine

# Warm up JAX (compiles kernels, loads CUDA/HIP)
SupportVectorMachine.warm_up()

# Build model
model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})

# Train
model.train(x=data_unit.train_x, y=data_unit.train_y)

# Predict (with_sign=False returns raw scores)
predict = model(data_unit.test_x)

# Save / Load
model.save("model_jax")
model = SupportVectorMachine.load_from("model_jax")
```

> Example code in [`example/normal_svm_jax.py`](./example/normal_svm_jax.py).
> Run on large datasets via [`run_JSVM.py`](./run_JSVM.py).

### SVM (NumPy / CPU Backend)

```python
from SVM import SupportVectorMachine

# Build model
model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})

# Train
model.train(x=data_unit.train_x, y=data_unit.train_y)

# Predict
predict = model(data_unit.test_x)

# Save / Load
model.save("model")
model = SupportVectorMachine.load_from("model")
```

> Example code in [`example/normal_svm.py`](./example/normal_svm.py).
> Run on large datasets via [`run_SVM.py`](./run_SVM.py).

---

## 📋 TODO

### ✅ Completed

- [x] Simple SVM (NumPy)
- [x] Simple Multi SVM (NumPy)
- [x] J-SVM (JAX binary SVM)
- [x] J-MultiSVM (JAX multi-class)
- [x] Load / Save for SVM
- [x] Load / Save for J-SVM
- [x] Run on iris dataset
- [x] Run on adult dataset

### 🔄 In Progress

- [ ] Load / Save for Multi SVM (NumPy)
- [ ] Load / Save for J-MultiSVM (JAX)

### 📌 Planned

- [ ] Test on additional datasets
- [ ] Parallel training (distributed SMO / data parallelism across GPUs)

> **Note:** Parallel feature is under active design. Contributions and ideas are welcome.

---

## 📐 Math: Nyström RBF Kernel Approximation

以 Nyström 近似 RBF kernel 的主要數學步驟：

1. 假設原始資料為 X ∈ ℝ^(N×D)，RBF kernel 為  
   K(xᵢ, xⱼ) = exp(-‖xᵢ - xⱼ‖² / (2σ²)).  
   此時完整的 Kernel 矩陣 K ∈ ℝ^(N×N)，元素為 Kᵢⱼ = K(xᵢ, xⱼ)。

2. 取 m (< N) 筆資料當作「錨點（landmarks）」，記為 Z ∈ ℝ^(m×D).  
   形成下列兩個子矩陣：  
   • K(X,Z): 形狀為 (N×m)，其元素為 Kᵢⱼ = K(xᵢ, zⱼ)  
   • K(Z,Z): 形狀為 (m×m)，其元素為 Kᵢⱼ = K(zᵢ, zⱼ)

3. 對 K(Z,Z) 做特徵分解或 SVD，例如：  
   K(Z,Z) = U S Uᵀ  
   其中 S 為對角矩陣，U 為對應的特徵向量矩陣。

4. 定義近似特徵映射 Φ(X)：  
   Φ(X) = K(X,Z) U S^(-1/2)  
   其中 S^(-1/2) 表示對角元素取 1/√sᵢ。

5. 對每個樣本 xᵢ，Nyström 近似對應的「低維特徵」為 Φ(xᵢ) 第 i 行，形狀約 (1×m)。  
   若想近似完整 NxN kernel，可用 Φ(X)·Φ(X)ᵀ 來取代原 K，使記憶體需求與運算量減少。

總結來說，Nyström 以「子集合」（landmarks）與矩陣分解，將高維或大規模 Kernel 矩陣轉換成低維度特徵，再以線性運算近似非線性 RBF，達到大幅節省記憶體、提升可擴充性的效果。

---

## 📚 API Reference

### `SupportVectorMachine`

The core binary SVM classifier, available in both `SVM` and `JSVM` modules.

#### Constructor

```python
model = SupportVectorMachine(
    C: float = 1.0,                          # Regularization parameter
    kernel_name: str = "linear",             # Kernel: "linear" | "rbf" | "poly"
    kernel_arg: dict | None = None,          # Kernel args, e.g. {"sigma": 2} for RBF
    threshold: float = 1e-5,                 # Support vector threshold
    approx_scale: int = 100                  # Nyström approximation rank (RBF only)
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `model.train(x, y)` | Train the SVM on data `x` (N×D) and labels `y` (N,) |
| `model(x, with_sign=False)` | Predict. `with_sign=True` returns ±1 class labels |
| `model.acc(x, y)` | Compute accuracy against ground truth |
| `model.save(path)` | Persist model to disk via joblib |
| `SupportVectorMachine.load_from(path)` | Load a saved model from disk |

### `MultiSupportVectorMachine`

One-vs-One multi-class wrapper. Same methods as `SupportVectorMachine`.

---

## 🔗 Reference

- **JAX** — [GitHub](https://github.com/jax-ml/jax) · [Video Tutorial](https://www.youtube.com/watch?v=_0D5lXDjNpw)
- **Jaxtyping** — [GitHub](https://github.com/patrick-kidger/jaxtyping)
- **Jaxopt** — [GitHub](https://jaxopt.github.io/stable/index.html)
- **Cvxpylayer** — [GitHub](https://github.com/cvxgrp/cvxpylayers)
- **awesome-jax** — [GitHub](https://github.com/n2cholas/awesome-jax?tab=readme-ov-file)

### Datasets

- **Pima Indians Diabetes Database** — [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Framingham Heart Study Dataset** — [Kaggle](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)
