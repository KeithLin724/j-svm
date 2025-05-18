# %%
# import os

# os.environ["JAX_PLATFORM_NAME"] = "cpu"

from sklearn import datasets
from sklearn.datasets import fetch_openml, fetch_kddcup99

from JSVM import SupportVectorMachine
from data import DataUnit, build_train_test_dataset
from rich import print
import os
from collections import Counter
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time

# %%
adult = fetch_openml(name="adult", version=2, as_frame=True)
X_adult, y_adult = adult.data, adult.target
print(f"Adult 資料集: 样本數 = {X_adult.shape[0]}, 特徵維度 = {X_adult.shape[1]}")
print(set(y_adult))
print(Counter(y_adult))

## %%
# 只取正負類各 1000 筆資料
N = 1000
pos_mask = y_adult == ">50K"
neg_mask = y_adult == "<=50K"

X_pos = X_adult[pos_mask].iloc[:N]
y_pos = y_adult[pos_mask].iloc[:N]

X_neg = X_adult[neg_mask].iloc[:N]
y_neg = y_adult[neg_mask].iloc[:N]

X_adult = pd.concat([X_pos, X_neg], axis=0).reset_index(drop=True)
y_adult = pd.concat([y_pos, y_neg], axis=0).reset_index(drop=True)

# %%
data = build_train_test_dataset(
    df_in=(X_adult, y_adult),
    train_size=0.5,
    positive_class=">50K",
    negative_class="<=50K",
    # label="income",
    # for_two_fold=True,
    return_data_unit=True,
    scaler=StandardScaler(),
)
# %%
print(data)
print(data.train_x.shape)
print(data.train_y.shape)
print(data.test_x.shape)
print(data.test_y.shape)


# %%
SupportVectorMachine.warm_up()
model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 0.5})

# %%
start = time.time()
model.train(data.train_x, data.train_y)
end = time.time()
print(model)
print(f"Training time: {end - start:.2f} seconds")

# %% forward time
start = time.time()
model(data.train_x)
end = time.time()
print(f"Forward time: {end - start:.2f} seconds")

# %%
train_acc, _ = model.acc(data.train_x, data.train_y)
print("Train accuracy:", train_acc)

test_acc, _ = model.acc(data.test_x, data.test_y)
print("Test accuracy:", test_acc)
# %%
# %%
# SupportVectorMachine.warm_up()
# model = SupportVectorMachine(
#     C=10, kernel_name="rbf", kernel_arg={"sigma": 0.5}, approx_scale=10
# )

# # %%
# start = time.time()
# model.train(data.train_x, data.train_y)
# end = time.time()
# print(model)
# print(f"Training time: {end - start:.2f} seconds")

# # %%
# train_acc, _ = model.acc(data.train_x, data.train_y)
# print("Train accuracy:", train_acc)

# test_acc, _ = model.acc(data.test_x, data.test_y)
# print("Test accuracy:", test_acc)

# %%
