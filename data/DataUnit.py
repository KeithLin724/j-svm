from dataclasses import dataclass, field
import pandas as pd
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

        train_y = label_to_index(train_y)
        test_y = label_to_index(test_y)
        return DataUnit(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            positive_class=positive_class,
            negative_class=negative_class,
            label_to_index=label_to_index,
            index_to_label=index_to_label,
        )
