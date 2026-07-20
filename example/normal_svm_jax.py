"""Binary SVM classification on the Iris dataset using the JAX backend."""
# %% imports
from JSVM import SupportVectorMachine
from data import IrisDataset, build_train_test_dataset, DataUnit
from utils import test_run, TestResult
from rich import print

# %% warm up
if __name__ == "__main__":
    SupportVectorMachine.warm_up()

# %% load data
if __name__ == "__main__":
    dataset = IrisDataset.load_iris_file(with_name=True)
    print(dataset.head())

    POSITIVE_CLASS = "Setosa"
    NEGATIVE_CLASS = "Virginica"

    pre_precess_data = build_train_test_dataset(
        df_in=(dataset.drop(columns=["Label"]), dataset["Label"]),
        train_size=IrisDataset.TRAIN_DATA_SIZE,
        positive_class=POSITIVE_CLASS,
        negative_class=NEGATIVE_CLASS,
    )
    print(pre_precess_data)

    data_unit = DataUnit.build_from_dict(
        data_dict=pre_precess_data,
        positive_class=POSITIVE_CLASS,
        negative_class=NEGATIVE_CLASS,
        label="Label",
    )
    print(data_unit)

# %% test model
if __name__ == "__main__":
    model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})

    output: list[TestResult] = test_run(
        data_unit=data_unit,
        kernel_name="rbf",
        kernel_arg={"sigma": [5, 1, 0.5, 0.1, 0.05]},
        C_list=[10],
        model_type=SupportVectorMachine,
        approx_scale=0,
    )
    print(output)

# %% train model
if __name__ == "__main__":
    model.train(x=data_unit.train_x, y=data_unit.train_y)
    model(data_unit.test_x)

# %% save
if __name__ == "__main__":
    model.save("dummy_model_jax")

# %% load model
if __name__ == "__main__":
    model = SupportVectorMachine.load_from("dummy_model_jax")
    print(model)
