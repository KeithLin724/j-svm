import numpy as np
import pytest

from SVM import MultiSupportVectorMachine
from data import IrisDataset


@pytest.fixture(scope="module")
def trained_multi_svm():
    df = IrisDataset.load_iris_file(with_name=True)
    model = MultiSupportVectorMachine(
        class_names=IrisDataset.LABEL,
        C=10,
        kernel_name="rbf",
        kernel_arg={"sigma": 2},
    )
    dataset = model.build_dataset(df, IrisDataset.TRAIN_DATA_SIZE)
    model.train(dataset["before"]["train"])
    return model, dataset


def test_multi_binary(trained_multi_svm):
    model, dataset = trained_multi_svm
    acc, _ = model.acc(dataset["before"]["test"])
    assert acc > 0.7


def test_multi_save_load(trained_multi_svm, tmp_path):
    model, dataset = trained_multi_svm
    x_test = dataset["before"]["test"].drop(columns=["Label"]).to_numpy()
    pred_orig = model(x_test)

    save_path = tmp_path / "multi_svm"
    model.save(str(save_path))
    loaded = MultiSupportVectorMachine.load_from(str(save_path))
    pred_loaded = loaded(x_test)

    np.testing.assert_array_equal(pred_orig, pred_loaded)
