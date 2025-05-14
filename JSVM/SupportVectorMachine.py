import cvxpy as cp
import numpy as np
from typing import Callable


def rbf(sigma: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def run(x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        # 检查维度是否匹配
        assert x_1.shape[0] == x_2.shape[0], "输入的两个向量必须有相同的特征数"

        # 计算欧氏距离的平方
        distance_squared = np.sum((x_1 - x_2) ** 2, axis=0)

        # 计算 RBF 核函数值
        kernel_value = np.exp(-distance_squared / (2 * sigma**2))

        if x_2.ndim != 1 and x_2.shape[1] == 1:
            kernel_value = kernel_value.reshape(-1, 1)

        return kernel_value

    return run


def poly(p: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def run(x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        assert x_1.shape[0] == x_2.shape[0], "输入的两个向量必须有相同的特征数"

        return (x_1.T @ x_2) ** p

    return run


def linear() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:

    def run(x_1: np.ndarray, x_2: np.ndarray) -> np.ndarray:
        assert x_1.shape[0] == x_2.shape[0], "输入的两个向量必须有相同的特征数"

        return x_1.T @ x_2

    return run


class Kernel:
    kernel_dict = {
        "linear": linear,
        "rbf": rbf,
        "poly": poly,
    }

    @staticmethod
    def get_kernel(
        name: str, config: dict
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if name not in Kernel.kernel_dict:
            raise NotImplementedError(
                f"{name} kernel not implemented. Available kernels: {list(Kernel.kernel_dict.keys())}"
            )

        func = Kernel.kernel_dict[name]

        return func(**config)


class SupportVectorMachine:
    def __init__(
        self,
        C: int,
        kernel_name: str = "linear",
        kernel_arg: dict = dict(),
        threshold: float = 1e-20,
    ):
        self._c = C
        self._threshold = threshold
        # like ("ay": ... , "x": ...,)
        self._a_y_x: np.ndarray = None
        self._b: float = None
        self._kernel = Kernel.get_kernel(kernel_name, kernel_arg)
        self._kernel_info = {"name": kernel_name, "arg": kernel_arg}
        return

    @property
    def alpha(self) -> np.ndarray:
        return self._a_y_x[:, 0].reshape(-1, 1)

    @property
    def bias(self) -> np.ndarray:
        return self._b

    def _build_k_matrix(self, x: np.ndarray) -> np.ndarray:
        size = x.shape[0]
        return np.array(
            [[self._kernel(x[i], x[j]) for j in range(size)] for i in range(size)]
        )

    def _find_alpha(
        self, K: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        size = x.shape[0]
        alpha = cp.Variable(size)

        big_k = np.outer(y, y) * K
        big_k = cp.psd_wrap(big_k)

        objective = cp.Minimize((1 / 2) * cp.quad_form(alpha, big_k) - cp.sum(alpha))
        constraints = [cp.sum(cp.multiply(alpha, y)) == 0, alpha >= 0, alpha <= self._c]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)  # verbose=True
        alpha = alpha.value

        # filter the important one
        support_vectors = np.where((self._c >= alpha) & (alpha > self._threshold))[0]
        return alpha, support_vectors

    def _find_bias(
        self,
        alpha: np.ndarray,
        support_vectors: np.ndarray,
        y: np.ndarray,
        K: np.ndarray,
    ):
        K_hat = K[support_vectors][:, support_vectors]
        alpha_hat = alpha[support_vectors]
        y_hat = y[support_vectors]

        res = y_hat - K_hat @ (alpha_hat * y_hat)

        return np.mean(res)

    def train(self, x: np.ndarray, y: np.ndarray):  # [batch, feature]   [batch, 1]
        # get the a
        K = self._build_k_matrix(x)
        # print(K)
        alpha, support_vector = self._find_alpha(K, x, y)
        # find the best b
        self._b = self._find_bias(alpha, support_vector, y, K)

        alpha, y = alpha.reshape(-1, 1), y.reshape(-1, 1)

        table = np.hstack((alpha, y, x))

        self._a_y_x = table[support_vector]

        return

    def cal_one_item(
        self, ay: np.ndarray, x_kernel: np.ndarray, x_item: np.ndarray
    ) -> np.ndarray:
        # x [1, feature]
        # x_kernel [a, feature]

        res = np.sum(ay * self._kernel(x_kernel.T, x_item.reshape(-1, 1))) + self._b

        return res

    def __call__(self, x: np.ndarray, with_sign: bool = False) -> np.ndarray:
        # x [batch, feature]
        a, y, x_kernel = (
            self._a_y_x[:, 0].reshape(-1, 1),
            self._a_y_x[:, 1].reshape(-1, 1),
            self._a_y_x[:, 2:],
        )

        result = [
            self.cal_one_item(a * y, x_kernel=x_kernel, x_item=x_item) for x_item in x
        ]
        res = np.hstack(result)

        if with_sign:
            res = np.sign(res)

        return res

    def acc(self, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:

        y_hat = self.__call__(x, True)

        return np.mean(y_hat == y), y_hat

    @property
    def info(self):
        return {
            "kernel": self._kernel_info,
            "support vector num": self._a_y_x.shape[0],
            "C": self._c,
            "b": f"{self._b:.4f}",
        }

    @property
    def short_name(self):
        kernel_name = self._kernel_info["name"]

        if len(self._kernel_info["arg"]) > 0:
            kernel_arg_name, kernel_arg = next(iter(self._kernel_info["arg"].items()))

        else:
            kernel_arg_name, kernel_arg = "", ""
        return f"{kernel_name}_{kernel_arg_name}_{kernel_arg}_C_{self._c}"

    def __str__(self):
        return str(self.info)

    def __repr__(self):
        return self.__str__()
