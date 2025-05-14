from pathlib import Path
import pandas as pd


class IrisDataset:

    LABEL = ["Setosa", "Versicolor", "Virginica"]

    COLOR_1 = dict(zip(LABEL, ["red", "green", "blue"]))
    COLOR_2 = dict(zip(LABEL, ["pink", "yellow", "orange"]))
    COLOR_3 = dict(zip(LABEL, ["brown", "lightgreen", "navy", "magenta"]))
    COLOR_4 = dict(zip(LABEL, ["teal", "gold", "violet", "coral"]))

    COLOR_SELECT = {
        "before": {
            "train": COLOR_1,
            "test": COLOR_2,
        },
        "after": {
            "train": COLOR_3,
            "test": COLOR_4,
        },
    }

    COLUMN_NAME = [
        "Sepal length",
        "Sepal width",
        "Petal length",
        "Petal width",
        "Label",
    ]
    TRAIN_DATA_SIZE = 25
    ASSETS = "./assets"

    def __init__(self):

        self._assets_folder = Path(IrisDataset.ASSETS)
        self._assets_folder.mkdir(parents=True, exist_ok=True)

        return

    @property
    def assets_folder(self):
        return self._assets_folder

    @staticmethod
    def load_iris_file(
        filename: str = "./iris.txt", with_name: bool = False
    ) -> pd.DataFrame:
        df = pd.read_fwf(filename)

        df_new = pd.DataFrame(
            {k: [v] for k, v in zip(IrisDataset.COLUMN_NAME, df.columns)}, dtype=float
        )
        df.columns = IrisDataset.COLUMN_NAME
        df_new = pd.concat([df_new, df], axis=0).reset_index().drop(columns=["index"])

        if not with_name:
            return df_new

        df_with_name = df_new.copy()

        df_with_name["Label"] = df_with_name["Label"].apply(
            lambda x: IrisDataset.LABEL[int(x) - 1]
        )

        return df_with_name
