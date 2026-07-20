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
![uv](https://img.shields.io/badge/uv-DE5FE9?style=for-the-badge&logo=uv&logoColor=white)

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
├── tests/                        # Unit tests (pytest)
│   ├── test_svm.py               # Binary SVM — NumPy backend
│   ├── test_jsvm.py              # Binary SVM — JAX backend
│   ├── test_multi_svm.py         # Multi-class SVM — NumPy backend
│   └── test_multi_jsvm.py        # Multi-class SVM — JAX backend
│
├── main.py                        # Entry point
├── run_JSVM.py                    # JAX runtime script
├── run_SVM.py                     # NumPy runtime script
└── pyproject.toml                 # Project configuration (uv)
```

---

## 🚀 Quick Start

### Installation

```sh
git clone https://github.com/KeithLin724/j-svm.git
cd j-svm
pip install uv
uv sync
uv run python -c "from JSVM import SupportVectorMachine; SupportVectorMachine.warm_up()"
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

See [TODO.md](./TODO.md) for project roadmap.

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
