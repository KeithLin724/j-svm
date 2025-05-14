import pandas as pd

from SVM import SupportVectorMachine as SVM
from JSVM import SupportVectorMachine as j_SVM

from data import DataUnit
from pathlib import Path

from typing import TypeVar
import numpy as np
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import time


@dataclass
class TestResult:
    model: SVM | j_SVM
    acc: float
    bias_: str
    alpha: np.ndarray
    training_time: float
    forward_time: float

    bias: float = field(repr=False)
    x_test: np.ndarray = field(repr=False)
    y_test: np.ndarray = field(repr=False)
    y_hat: np.ndarray = field(repr=False)

    @classmethod
    def build(
        cls, model: SVM | j_SVM, x: np.ndarray, y: np.ndarray, training_time: float
    ):
        model = model
        acc, y_hat = model.acc(x=x, y=y)
        alpha = model.alpha.reshape(-1)
        bias = model.bias
        bias_ = f"{model.bias:.4f}"

        start = time.time()
        model(x, with_sign=True)
        end = time.time()
        forward_time = end - start

        return cls(
            model, acc, bias_, alpha, training_time, forward_time, bias, x, y, y_hat
        )

    def to_data_dict(self, image_save_folder: Path = None) -> dict:
        data = {
            "model_name": str(self.model.short_name),
            "acc": f"{self.acc:.2f}",
            "alpha": str([f"{item:.4f}" for item in self.alpha.tolist()]),
            "bias": self.bias_,
        }

        if image_save_folder is not None:
            filename = image_save_folder.joinpath(f"{self.model.short_name}.png")
            data |= {"image": f"![alt]({filename})"}

        return data

    def plot_decision_boundary(
        self, save_folder: Path = None, resolution=100, padding=1.0, with_contours=True
    ):
        """
        绘制任何二维模型的决策边界。

        Parameters:
        - model: 训练好的模型，必须支持 `predict` 或类似接口 (可以传入自定义SVM模型)。
        - x: 训练数据，形状为 [num_samples, 2]。
        - y: 训练标签，形状为 [num_samples] 或 [num_samples, 1]。
        - resolution: 网格的分辨率，默认为 100。
        - padding: 数据边界填充距离，以保证决策边界的完整显示。
        - with_contours: 是否显示轮廓线，默认为 True。
        """
        # 设定网格范围
        x, y = self.x_test, self.y_test
        x_min, x_max = x[:, 0].min() - padding, x[:, 0].max() + padding
        y_min, y_max = x[:, 1].min() - padding, x[:, 1].max() + padding
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # 在网格点上获取模型的预测值
        Z = self.model(grid_points, with_sign=True)

        Z = Z.reshape(xx.shape)

        # 绘制训练数据点和决策边界
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap="bwr", s=50, edgecolors="k")
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")

        if with_contours:
            plt.contour(
                xx, yy, Z, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors="k"
            )

        # 添加图例和标签
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title(
            f"Model decision boundary\nModel:{self.model}\nAcc:{self.acc*100:.2f}%"
        )

        if save_folder is not None:
            filename = save_folder.joinpath(f"{self.model.short_name}.png")
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.1)

        plt.show()


def test_run(
    data_unit: DataUnit,
    kernel_name: str,
    kernel_arg: dict = dict(),
    C_list: list[int] = [1, 10, 100],
    model_type: SVM | j_SVM = SVM,
):

    if len(kernel_arg) != 0:
        kernel_item = next(iter(kernel_arg.items()))
        kernel_arg_name, kernel_arg_value_list = kernel_item
        kernel_arg_list = [{kernel_arg_name: item} for item in kernel_arg_value_list]
    else:
        kernel_arg_list = [dict()]
    # return
    train_x, train_y = data_unit.train_x, data_unit.train_y
    test_x, test_y = data_unit.test_x, data_unit.test_y

    model_list = []

    for kernel_arg_item in kernel_arg_list:
        for c in C_list:
            model = model_type(C=c, kernel_name=kernel_name, kernel_arg=kernel_arg_item)
            start = time.time()
            model.train(x=train_x, y=train_y)
            end = time.time()
            training_time = end - start

            result = TestResult.build(
                model, x=test_x, y=test_y, training_time=training_time
            )

            model_list.append(result)

    return model_list
