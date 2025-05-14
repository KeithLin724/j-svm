# %% load libraries
from SVM import MultiSupportVectorMachine
from data import IrisDataset, build_train_test_dataset, DataUnit

# %% sample dataset
dataset = IrisDataset.load_iris_file(with_name=True)

# %% peek dataset
print(dataset.head())

# %% multi-class classification
model = MultiSupportVectorMachine(
    class_names=IrisDataset.LABEL,
    C=10,
    kernel_name="rbf",
    kernel_arg={"sigma": 2},
)


# %%pre-process data
train_data, test_data = (
    dataset[: IrisDataset.TRAIN_DATA_SIZE],
    dataset[IrisDataset.TRAIN_DATA_SIZE :],
)


# %% Train model
each_model_training_acc = model.train(train_data)

# test
acc, predict = model.acc(test_data)


# %%
print(f"Training accuracy: {each_model_training_acc}")
print(f"Testing accuracy: {acc}")
print(f"Testing predict: {predict}")
