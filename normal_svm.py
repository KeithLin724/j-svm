# %% load libraries
from SVM import SupportVectorMachine
from data import IrisDataset, build_train_test_dataset, DataUnit

# %% sample dataset
dataset = IrisDataset.load_iris_file(with_name=True)

# %% peek dataset
print(dataset.head())

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
)

print(pre_precess_data)

# %%
import numpy as np

label_to_index = {POSITIVE_CLASS: 1, NEGATIVE_CLASS: -1}
index_to_label = {1: POSITIVE_CLASS, -1: NEGATIVE_CLASS}


label_to_index_np = np.vectorize(label_to_index.get)
index_to_label_np = np.vectorize(index_to_label.get)

# %% Build DataUnit
data_unit = DataUnit.build_from_dict(
    data_dict=pre_precess_data,
    label_to_index=label_to_index_np,
    index_to_label=index_to_label_np,
    label="Label",
)
print(data_unit)
# %% test model
from utils import test_run, TestResult

output = test_run(
    data_unit=data_unit,
    kernel_name="rbf",
    kernel_arg={"sigma": [5, 1, 0.5, 0.1, 0.05]},
    C_list=[10],
    model_type=SupportVectorMachine,
)
# %% show result
print(output)

# train model
# model.train(x=data_unit.train_x, y=data_unit.train_y)
