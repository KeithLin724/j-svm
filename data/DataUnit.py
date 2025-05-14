from dataclasses import dataclass, field
import jax
import numpy as np


@dataclass(slots=True)
class DataUnit:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray

    positive_class: str
    negative_class: str

    label_to_index: np.vectorize = field(repr=False, default=None)
    index_to_label: np.vectorize = field(repr=False, default=None)

    @staticmethod
    def build_from_dict(
        data_dict: dict,
        positive_class: str,
        negative_class: str,
        label: str = "Label",
    ):
        train_data = data_dict["train"]
        test_data = data_dict["test"]
        train_x, train_y = (
            train_data.drop(columns=[label]).to_numpy(),
            train_data[label].to_numpy(),
        )
        test_x, test_y = (
            test_data.drop(columns=[label]).to_numpy(),
            test_data[label].to_numpy(),
        )

        label_to_index = {positive_class: 1, negative_class: -1}
        index_to_label = {1: positive_class, -1: negative_class}

        label_to_index_np = np.vectorize(label_to_index.get)
        index_to_label_np = np.vectorize(index_to_label.get)

        train_y = label_to_index_np(train_y)
        test_y = label_to_index_np(test_y)
        return DataUnit(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            positive_class=positive_class,
            negative_class=negative_class,
            label_to_index=label_to_index_np,
            index_to_label=index_to_label_np,
        )

    def __or__(self, other):
        if not isinstance(other, DataUnit):
            return NotImplemented

        assert (
            self.positive_class == other.positive_class
            and self.negative_class == other.negative_class
        ), "DataUnit must have the same positive and negative classes to be merged."

        train_x = np.concatenate([self.train_x, other.train_x], axis=0)
        train_y = np.concatenate([self.train_y, other.train_y], axis=0)
        test_x = np.concatenate([self.test_x, other.test_x], axis=0)
        test_y = np.concatenate([self.test_y, other.test_y], axis=0)

        return DataUnit(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            positive_class=self.positive_class,
            negative_class=self.negative_class,
            label_to_index=self.label_to_index,
            index_to_label=self.index_to_label,
        )

    def to_devices(self, device):
        self.train_x = jax.device_put(self.train_x, device)
        self.train_y = jax.device_put(self.train_y, device)
        self.test_x = jax.device_put(self.test_x, device)
        self.test_y = jax.device_put(self.test_y, device)
