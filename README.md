# j-SVM: GPU-Accelerated Parallel SVM using JAX

> ⭐ **If you find this project helpful, please give it a star on GitHub!**

## Platform

![nVIDIA](https://img.shields.io/badge/cuda-000000.svg?style=for-the-badge&logo=nVIDIA&logoColor=green) ![AMD_HIP](https://img.shields.io/badge/HIP-%23000000.svg?style=for-the-badge&logo=amd&logoColor=white&logoSize=auto)

## Tools

![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Poetry](https://img.shields.io/badge/Poetry-%233B82F6.svg?style=for-the-badge&logo=poetry&logoColor=0B3D8D)

## System

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

## TODO

- [x] Complete Simple SVM
- [x] Complete Simple Multi SVM
- [x] Complete J-SVM
- [x] Complete J-MultiSVM
- [x] Add Load Save for SVM
- [x] Add Load Save for J-SVM
- [ ] Add Load Save for Multi SVM
- [ ] Add Load Save for J-MultiSVM
- [ ] Test on different dataset
  - [x] run on iris dataset
  - [x] run on adult dataset
- [ ] Add parallel feature

## Build Env

```sh
pip install poetry 

poetry install 
```

---

## How to use J-SVM and SVM

### J-SVM

```python
from JSVM import SupportVectorMachine

# warm up j_SVM
SupportVectorMachine.warm_up()

# build model 
model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})

# train model 
model.train(x=data_unit.train_x, y=data_unit.train_y)

# model predict  
predict = model(data_unit.test_x)

# save model 
model.save("model_jax")

# load model 
model = SupportVectorMachine.load_from("model_jax")
```

> Example code in [`example/normal_svm_jax.py`](./example/normal_svm_jax.py),
> Run in large dataset ['run_JSVM.py'](./run_JSVM.py)

### SVM

```python
from SVM import SupportVectorMachine

# build model 
model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})

# train model 
model.train(x=data_unit.train_x, y=data_unit.train_y)

# model predict  
predict = model(data_unit.test_x)

# save model 
model.save("model")

# load model 
model = SupportVectorMachine.load_from("model")
```

> Example code in [`example/normal_svm.py`](./example/normal_svm.py),
> Run in large dataset ['run_SVM.py'](./run_SVM.py)

---

## Math

以 Nystroem 近似 RBF kernel 的主要數學步驟：

1. 假設原始資料為 X ∈ ℝ^(N×D)，RBF kernel 為  
   K(xᵢ, xⱼ) = exp(-‖xᵢ - xⱼ‖² / (2σ²))。  
   此時完整的 Kernel 矩陣 K ∈ ℝ^(N×N)，元素為 Kᵢⱼ = K(xᵢ, xⱼ)。

2. 取 m (< N) 筆資料當作「錨點（landmarks）」，記為 Z ∈ ℝ^(m×D)。  
   形成下列兩個子矩陣：  
   • K(X,Z): 形狀為 (N×m)，其元素為 Kᵢⱼ = K(xᵢ, zⱼ)  
   • K(Z,Z): 形狀為 (m×m)，其元素為 Kᵢⱼ = K(zᵢ, zⱼ)

3. 對 K(Z,Z) 做特徵分解或 SVD，例如：  
   K(Z,Z) = U S Uᵀ  
   其中 S 為對角矩陣，U 為對應的特徵向量矩陣。

4. 定義近似特徵映射 Φ(X)：  
   Φ(X) = K(X,Z) U S^(-1/2)  
   其中 S^(-1/2) 表示對角元素取 1/√sᵢ。

5. 對每個樣本 xᵢ，Nystroem 近似對應的「低維特徵」為 Φ(xᵢ) 第 i 行，形狀約 (1×m)。  
   若想近似完整 NxN kernel，可用 Φ(X)·Φ(X)ᵀ 來取代原 K，使記憶體需求與運算量減少。

總結來說，Nystroem 以「子集合」（landmarks）與矩陣分解，將高維或大規模 Kernel 矩陣轉換成低維度特徵，再以線性運算近似非線性 RBF，達到大幅節省記憶體、提升可擴充性的效果。

---

## Reference

Jax : [here](https://github.com/jax-ml/jax),[video](https://www.youtube.com/watch?v=_0D5lXDjNpw)

JaxTyping : [here](https://github.com/patrick-kidger/jaxtyping)

Jaxopt : [here](https://jaxopt.github.io/stable/index.html)

Cvxpylayer : [here](https://github.com/cvxgrp/cvxpylayers)

awesome-jax : [here](https://github.com/n2cholas/awesome-jax?tab=readme-ov-file)

## Dataset

Pima Indians Diabetes Database : [here](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Framingham heart study dataset : [here](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset)
