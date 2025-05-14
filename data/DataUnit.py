from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class DataUnit:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray

    label_to_index: np.vectorize = field(repr=False, default=None)
    index_to_label: np.vectorize = field(repr=False, default=None)

    @staticmethod
    def build_from_dict(
        data_dict: dict,
        label_to_index: np.vectorize,
        index_to_label: np.vectorize,
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
        train_y = label_to_index(train_y)
        test_y = label_to_index(test_y)
        return DataUnit(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            label_to_index=label_to_index,
            index_to_label=index_to_label,
        )
