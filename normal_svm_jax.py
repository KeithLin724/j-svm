# %%
# import os

# os.environ["JAX_PLATFORM_NAME"] = "cpu"


# %% load libraries
from JSVM import SupportVectorMachine
from data import IrisDataset, build_train_test_dataset, DataUnit

# %% sample dataset
dataset = IrisDataset.load_iris_file(with_name=True)

# %% peek dataset
print(dataset.head())

# %% warm up
SupportVectorMachine.warm_up()

# %% binary classification
model = SupportVectorMachine(C=10, kernel_name="rbf", kernel_arg={"sigma": 2})

# %% load dataset
POSITIVE_CLASS = "Setosa"
NEGATIVE_CLASS = "Virginica"

pre_precess_data = build_train_test_dataset(
    df_in=dataset,
    train_size=IrisDataset.TRAIN_DATA_SIZE,
    positive_class=POSITIVE_CLASS,
    negative_class=NEGATIVE_CLASS,
    to_one_hot=False,
)

print(pre_precess_data)


# %% Build DataUnit
data_unit = DataUnit.build_from_dict(
    data_dict=pre_precess_data,
    positive_class=POSITIVE_CLASS,
    negative_class=NEGATIVE_CLASS,
    label="Label",
)
print(data_unit)
# %% test model
from utils import test_run, TestResult

output: list[TestResult] = test_run(
    data_unit=data_unit,
    kernel_name="rbf",
    kernel_arg={"sigma": [5, 1, 0.5, 0.1, 0.05]},
    C_list=[10],
    model_type=SupportVectorMachine,
    use_approx=True,
)
# %% show result
print(output)

# %% train model
model.train(x=data_unit.train_x, y=data_unit.train_y)

# %%

# data_unit.train_x = jax.device_put(data_unit.train_x)

model(data_unit.test_x)


# %% save
model.save("dummy_model_jax")

# %% load model
model = SupportVectorMachine.load_from("dummy_model_jax")
print(model)

# %%
