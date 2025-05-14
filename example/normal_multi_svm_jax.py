# %% load libraries
from JSVM import MultiSupportVectorMachine
from data import IrisDataset, build_train_test_dataset, DataUnit
import time


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


# # %%pre-process data
# train_data, test_data = (
#     dataset[: IrisDataset.TRAIN_DATA_SIZE],
#     dataset[IrisDataset.TRAIN_DATA_SIZE :],
# )


# %% build data set
dataset = model.build_dataset(dataset, IrisDataset.TRAIN_DATA_SIZE)

# %% Train model
start = time.time()
each_model_training_acc = model.train(dataset["before"]["train"])
end = time.time()
print(f"Training time: {end - start:.2f} seconds")


data_in = dataset["before"]["test"].drop(columns=["Label"]).to_numpy()

start = time.time()
model(data_in)
end = time.time()
print(f"Forward time: {end - start:.2f} seconds")

# test
acc, predict = model.acc(dataset["before"]["test"])


# %%
print(f"Training accuracy: {each_model_training_acc}")
print(f"Testing accuracy: {acc}")
print(f"Testing predict: {predict}")

# %%
