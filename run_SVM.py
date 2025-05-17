# %%
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"


from sklearn import datasets
from sklearn.datasets import fetch_openml, fetch_kddcup99

from JSVM import SupportVectorMachine
from data import DataUnit, build_train_test_dataset
from rich import print
import os

# %%
adult = fetch_openml(name="adult", version=2, as_frame=True)
X_adult, y_adult = adult.data, adult.target
print(f"Adult 資料集: 样本數 = {X_adult.shape[0]}, 特徵維度 = {X_adult.shape[1]}")
print(set(y_adult))

# %%
data = build_train_test_dataset(
    df_in=(X_adult, y_adult),
    train_size=0.8,
    positive_class=">50K",
    negative_class="<=50K",
    # label="income",
    # for_two_fold=True,
    return_data_unit=True,
)
# %%
print(data)
print(data.train_x.shape)
print(data.train_y.shape)
print(data.test_x.shape)
print(data.test_y.shape)


# %%
# SupportVectorMachine.warm_up()
model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})

# %%
model.train(data.train_x, data.train_y)
print(model)

# %%
