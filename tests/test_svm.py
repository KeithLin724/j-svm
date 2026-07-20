import numpy as np
import pytest
from sklearn import datasets
from sklearn.model_selection import train_test_split

from SVM import SupportVectorMachine


@pytest.fixture(scope="module")
def binary_iris():
    """Prepare binary iris data (Setosa vs Versicolor) with ±1 labels."""
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    mask = y < 2
    X_bin, y_bin = X[mask], y[mask]
    y_bin = y_bin.astype(np.float64) * 2 - 1  # {0, 1} -> {-1, 1}
    return train_test_split(X_bin, y_bin, test_size=0.3, random_state=42)


def test_binary_classification(binary_iris):
    X_train, X_test, y_train, y_test = binary_iris
    model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})
    model.train(X_train, y_train)
    acc, _ = model.acc(X_test, y_test)
    assert acc > 0.8


def test_save_load(binary_iris, tmp_path):
    X_train, X_test, y_train, y_test = binary_iris
    model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})
    model.train(X_train, y_train)
    pred_orig = model(X_test, with_sign=True)

    save_path = tmp_path / "svm_model"
    model.save(str(save_path))
    loaded = SupportVectorMachine.load_from(str(save_path))
    pred_loaded = loaded(X_test, with_sign=True)

    np.testing.assert_array_equal(pred_orig, pred_loaded)


def test_kernel_rbf(binary_iris):
    X_train, X_test, y_train, y_test = binary_iris
    model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})
    model.train(X_train, y_train)  # should not raise


def test_kernel_linear(binary_iris):
    X_train, X_test, y_train, y_test = binary_iris
    model = SupportVectorMachine(C=10, kernel_name="linear")
    model.train(X_train, y_train)  # should not raise
