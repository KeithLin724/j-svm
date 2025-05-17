import pandas as pd
from typing import Any

from .DataUnit import DataUnit


def process_data_to_one_hot(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical columns to one-hot encoding.
    """
    # Identify categorical columns
    categorical_cols = df_in.select_dtypes(include=["object", "category"]).columns

    # Apply one-hot encoding
    df_out = pd.get_dummies(df_in, columns=categorical_cols, drop_first=True)

    # Ensure all one-hot encoded columns are 0 or 1 (not True/False)
    df_out = df_out.astype(int)

    return df_out


def build_train_test_dataset(
    df_in: pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame],
    train_size: int | float,
    positive_class: Any,
    negative_class: Any,
    label: str = "Label",
    for_two_fold: bool = False,
    return_data_unit: bool = False,
    to_one_hot: bool = True,
):
    if to_one_hot:
        if isinstance(df_in, tuple):
            df_in_0 = process_data_to_one_hot(df_in[0])
            df_in_0[label] = df_in[1]
            df_in = df_in_0
        else:
            df_in = process_data_to_one_hot(df_in)

    # have before and after -> for two-fold
    positive_data = df_in[df_in[label] == positive_class]
    negative_data = df_in[df_in[label] == negative_class]

    if isinstance(train_size, float) and 0 < train_size < 1:
        train_size_pos = int(len(positive_data) * train_size)
        train_size_neg = int(len(negative_data) * train_size)
    else:
        train_size_pos = int(train_size)
        train_size_neg = int(train_size)

    before = [positive_data[:train_size_pos], negative_data[:train_size_neg]]

    after = [positive_data[train_size_pos:], negative_data[train_size_neg:]]

    before, after = pd.concat(before), pd.concat(after)

    if for_two_fold:
        res = {
            "before": {
                "train": before,
                "test": after,
            },
            "after": {
                "train": after,
                "test": before,
            },
        }

        if return_data_unit:
            res["before"] = DataUnit.build_from_dict(
                data_dict=res["before"],
                positive_class=positive_class,
                negative_class=negative_class,
                label=label,
            )
            res["after"] = DataUnit.build_from_dict(
                data_dict=res["after"],
                positive_class=positive_class,
                negative_class=negative_class,
                label=label,
            )

    else:
        res = {
            "train": before,
            "test": after,
        }
        if return_data_unit:
            res = DataUnit.build_from_dict(
                data_dict=res,
                positive_class=positive_class,
                negative_class=negative_class,
                label=label,
            )

    return res
