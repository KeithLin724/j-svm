import numpy as np
import jax.numpy as jnp

from dataclasses import dataclass, asdict
from pathlib import Path


import json


@dataclass(slots=True)
class SVMParameter:
    C: int
    threshold: float
    kernel_name: str
    kernel_arg: dict

    ## after training
    a_y_x: np.ndarray | jnp.ndarray | None = None
    b: float | None = None

    ## override

    def save(self, path: str | Path):

        data = asdict(self)

        if isinstance(path, str):
            path = Path(path)

        path.mkdir(parents=True, exist_ok=True)

        parameter_path = path / "parameter.json"
        matrix_path = path / "matrix.npy"

        if self.a_y_x is not None:

            if isinstance(self.a_y_x, jnp.ndarray):
                self.a_y_x = np.asarray(self.a_y_x)

            np.save(matrix_path, self.a_y_x)
            data["a_y_x"] = str(matrix_path)

        else:
            data["a_y_x"] = None

        if self.b is not None:
            data["b"] = float(self.b)

        with open(parameter_path, "w") as f:
            json.dump(data, f, indent=4)

        return

    @staticmethod
    def load_from(path: str | Path):

        if isinstance(path, str):
            path = Path(path)

        assert path.exists(), f"Path {path} does not exist"

        parameter_path = path / "parameter.json"
        # matrix_path = path / "matrix.npz"

        with open(parameter_path, "r") as f:
            data = json.load(f)

        if data["a_y_x"] is not None:
            data["a_y_x"] = Path(data["a_y_x"])

            if not data["a_y_x"].exists():
                raise FileNotFoundError(f"Matrix file {data['a_y_x']} does not exist")

            data["a_y_x"] = np.load(data["a_y_x"])

        return SVMParameter(**data)
