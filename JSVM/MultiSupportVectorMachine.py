from itertools import combinations
from dataclasses import dataclass, field

import numpy as np
import jax.numpy as jnp
import pandas as pd

from .SupportVectorMachine import SupportVectorMachine

# from joblib import Parallel, delayed


@dataclass(slots=True)
class SvmModelModule:
    pair: tuple[str, str]
    model: SupportVectorMachine
    positive_class: str
    negative_class: str

    # for mapping
    label_to_index_np: np.vectorize = field(repr=False)
    index_to_label_np: np.vectorize = field(repr=False)

    label: str = field(default="Label", repr=False)

    @classmethod
    def build(
        cls,
        pair: tuple[str, str],
        C: int,
        kernel_name: str,
        kernel_arg: dict,
        threshold: float,
        label: str = "Label",
    ):
        positive_class, negative_class = pair
        model = SupportVectorMachine(
            C=C,
            kernel_name=kernel_name,
            kernel_arg=kernel_arg,
            threshold=threshold,
        )

        label_to_index = {positive_class: 1, negative_class: -1}
        index_to_label = {1: positive_class, 0: "idk", -1: negative_class}

        label_to_index_np = np.vectorize(label_to_index.get)
        index_to_label_np = np.vectorize(index_to_label.get)

        return cls(
            pair,
            model,
            positive_class,
            negative_class,
            label_to_index_np,
            index_to_label_np,
            label=label,
        )

    def build_train_test_dataset(self, df_in: pd.DataFrame, train_size: int):
        # have before and after -> for two-fold
        positive_data = df_in[df_in[self.label] == self.positive_class]
        negative_data = df_in[df_in[self.label] == self.negative_class]

        before = [positive_data[:train_size], negative_data[:train_size]]
        after = [positive_data[train_size:], negative_data[train_size:]]

        return {
            "before": {
                "train": pd.concat(before),
                "test": pd.concat(after),
            },
            "after": {
                "train": pd.concat(after),
                "test": pd.concat(before),
            },
        }

    def build_for_model_input(
        self, df_in: pd.DataFrame
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        in_x, in_y = (
            df_in.drop(columns=[self.label]).to_numpy(),
            df_in[self.label].to_numpy(),
        )

        in_y = self.label_to_index_np(in_y)
        return in_x, in_y

    # need to cut the data
    def train(self, df_in: pd.DataFrame):
        x, y = self.build_for_model_input(df_in=df_in)

        self.model.train(x, y)
        return

    # for the analysis
    def acc(self, df_in: pd.DataFrame) -> float:
        x, y = self.build_for_model_input(df_in=df_in)
        acc = self.model.acc(x, y)
        return acc

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        res = self.model(x, with_sign=True)

        res_label = self.index_to_label_np(res)

        return res_label


class MultiSupportVectorMachine:

    STATES = ["before", "after"]

    def __init__(
        self,
        class_names: list[str],
        C: int,
        kernel_name: str = "rbf",
        kernel_arg: dict = dict(),
        threshold: float = 1e-20,
        label: str = "Label",
    ):
        self._C = C
        self._kernel_name = kernel_name
        self._kernel_arg = kernel_arg
        self._threshold = threshold
        self._class_names = class_names

        class_combination = list(combinations(class_names, 2))
        # print(f"Class combinations: {class_combination}")

        # build model
        self._models = {
            pair: SvmModelModule.build(
                pair=pair,
                C=self._C,
                kernel_name=self._kernel_name,
                kernel_arg=self._kernel_arg,
                threshold=self._threshold,
                label=label,
            )
            for pair in class_combination
        }

        return

    @staticmethod
    def warm_up():
        SupportVectorMachine.warm_up()
        return

    def __repr__(self):
        return f"class name: {self._class_names}, C: {self._C}, kernel: {self._kernel_name} ({self._kernel_arg})"

    def train(self, data_in: dict[tuple[str, str], pd.DataFrame]) -> dict:
        acc_dict = dict()

        for pair, training_data in data_in.items():
            self._models[pair].train(training_data)

            acc_dict[pair] = self._models[pair].acc(training_data)

        return acc_dict

    def _get_most_freq_by_row(self, row: jnp.ndarray):
        unique, counts = np.unique(row, return_counts=True)
        return unique[np.argmax(counts)]

    def predict(self, x: np.ndarray) -> np.ndarray:

        res = [model.predict(x) for model in self._models.values()]
        res_np = np.array(res).T
        res = np.array([self._get_most_freq_by_row(row) for row in res_np])
        return res

    def __call__(self, x: jnp.ndarray):
        return self.predict(x)

    def acc(self, df_in: pd.DataFrame) -> tuple[float, jnp.ndarray]:
        x, y = df_in.drop(columns=["Label"]).to_numpy(), df_in["Label"].to_numpy()
        res = self.predict(x)

        return jnp.mean(res == y), res

    def build_dataset(self, df_in: pd.DataFrame, training_size: int) -> dict:
        dataset = {
            "before": {"train": dict(), "test": []},
            "after": {"train": dict(), "test": []},
        }

        # build dataset
        for pair in self._models.keys():
            part_of_dataset = self._models[pair].build_train_test_dataset(
                df_in, train_size=training_size
            )

            for state in MultiSupportVectorMachine.STATES:
                dataset[state]["train"][pair] = part_of_dataset[state]["train"]
                dataset[state]["test"].append(part_of_dataset[state]["test"])

        # merge to same dataset "test"
        for state in MultiSupportVectorMachine.STATES:
            dataset[state]["test"] = pd.concat(dataset[state]["test"])

        return dataset
