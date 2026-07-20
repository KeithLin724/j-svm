"""Multi-class SVM (One-vs-One) on the Iris dataset using the NumPy backend."""
# %% imports
from SVM import MultiSupportVectorMachine
from data import IrisDataset

# %% load data
if __name__ == "__main__":
    dataset = IrisDataset.load_iris_file(with_name=True)
    print(dataset.head())

# %% multi-class classification
if __name__ == "__main__":
    model = MultiSupportVectorMachine(
        class_names=IrisDataset.LABEL,
        C=10,
        kernel_name="rbf",
        kernel_arg={"sigma": 2},
    )

# %% build data set
if __name__ == "__main__":
    dataset = model.build_dataset(dataset, IrisDataset.TRAIN_DATA_SIZE)

# %% Train model
if __name__ == "__main__":
    each_model_training_acc = model.train(dataset["before"]["train"])

# %% test
if __name__ == "__main__":
    acc, predict = model.acc(dataset["before"]["test"])
    print(f"Training accuracy: {each_model_training_acc}")
    print(f"Testing accuracy: {acc}")
    print(f"Testing predict: {predict}")
