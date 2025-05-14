import pandas as pd
from typing import Any

from .DataUnit import DataUnit


def build_train_test_dataset(
    df_in: pd.DataFrame,
    train_size: int,
    positive_class: Any,
    negative_class: Any,
    label: str = "Label",
    for_two_fold: bool = False,
    return_data_unit: bool = False,
):
    # have before and after -> for two-fold
    positive_data = df_in[df_in[label] == positive_class]
    negative_data = df_in[df_in[label] == negative_class]

    before = [positive_data[:train_size], negative_data[:train_size]]

    after = [positive_data[train_size:], negative_data[train_size:]]

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
