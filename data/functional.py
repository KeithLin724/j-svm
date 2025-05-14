import pandas as pd
from typing import Any


def build_train_test_dataset(
    df_in: pd.DataFrame,
    train_size: int,
    positive_class: Any,
    negative_class: Any,
    for_two_fold: bool = False,
):
    # have before and after -> for two-fold
    positive_data = df_in[df_in["Label"] == positive_class]
    negative_data = df_in[df_in["Label"] == negative_class]

    before = [positive_data[:train_size], negative_data[:train_size]]

    after = [positive_data[train_size:], negative_data[train_size:]]

    if for_two_fold:
        res = {
            "before": {
                "train": pd.concat(before),
                "test": pd.concat(after),
            },
            "after": {
                "train": pd.concat(after),
                "test": pd.concat(before),
            },
        }

    # else
    return {
        "train": pd.concat(before),
        "test": pd.concat(after),
    }
