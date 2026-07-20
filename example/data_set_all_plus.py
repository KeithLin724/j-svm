"""Load and inspect multiple benchmark datasets."""
# %% imports
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml, fetch_kddcup99

# %% small datasets
if __name__ == "__main__":
    # 1. Iris
    iris = datasets.load_iris()
    X_iris, y_iris = iris.data, iris.target
    print(f"Iris 資料集: 样本數 = {X_iris.shape[0]}, 特徵維度 = {X_iris.shape[1]}")
    print(f"特徵名稱: {iris.feature_names}")

    # 2. Wine
    wine = datasets.load_wine()
    X_wine, y_wine = wine.data, wine.target
    print(f"Wine 資料集: 样本數 = {X_wine.shape[0]}, 特徵維度 = {X_wine.shape[1]}")
    print(f"特徵名稱: {wine.feature_names}")

    # 3. Breast Cancer
    cancer = datasets.load_breast_cancer()
    X_cancer, y_cancer = cancer.data, cancer.target
    print(
        f"Breast Cancer 資料集: 样本數 = {X_cancer.shape[0]}, 特徵維度 = {X_cancer.shape[1]}"
    )
    print(f"特徵名稱: {cancer.feature_names}")

# %% medium and large datasets
if __name__ == "__main__":
    # 4. Adult (Census Income)
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X_adult, y_adult = adult.data, adult.target
    print(f"Adult 資料集: 样本數 = {X_adult.shape[0]}, 特徵維度 = {X_adult.shape[1]}")
    print(f"Adult 資料集: 標籤數 = {len(set(y_adult))}")

    # 5. MNIST
    dataset_name = "mnist_784"
    mnist = fetch_openml(name=dataset_name, version=1)
    X_mnist, y_mnist = mnist.data, mnist.target
    print(f"MNIST 資料集: 样本數 = {X_mnist.shape[0]}, 特徵維度 = {X_mnist.shape[1]}")
    print(f"MNIST 資料集: 標籤數 = {len(set(y_mnist))}")

    # 6. 20 Newsgroups
    twenty = datasets.fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )
    X_20news, y_20news = twenty.data, twenty.target
    print(f"20 Newsgroups 資料集: 文件數 = {len(X_20news)}")
    print(f"20 Newsgroups 資料集: 標籤數 = {len(set(y_20news))}")

    # 7. KDD Cup 1999
    kdd = fetch_kddcup99(percent10=True)
    X_kdd, y_kdd = kdd.data, kdd.target
    print(f"KDD Cup 1999 10%子集: 样本數 = {X_kdd.shape[0]}, 特徵維度 = {X_kdd.shape[1]}")
    print(f"KDD Cup 1999 10%子集: 標籤類別 = {set(y_kdd)}")

    # 8. HIGGS
    higgs = fetch_openml(name="HIGGS", version=1)
    X_higgs, y_higgs = higgs.data, higgs.target
    print(f"HIGGS 資料集: 样本數 = {X_higgs.shape[0]}, 特徵維度 = {X_higgs.shape[1]}")
    print(f"HIGGS 資料集: 標籤類別 = {set(y_higgs)}")
