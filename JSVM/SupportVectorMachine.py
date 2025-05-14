import jax
import jax.numpy as jnp
import numpy as np
import cvxpy as cp
from jaxtyping import Array, Float, Int
from typing import Callable

# from cvxpylayers.torch import CvxpyLayer


def rbf(sigma: float) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

    @jax.jit
    def rbf_jit(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        # 计算欧氏距离的平方
        distance_squared = jnp.sum((x_1 - x_2) ** 2, axis=0)

        # 计算 RBF 核函数值
        kernel_value = jnp.exp(-distance_squared / (2 * sigma**2))

        return kernel_value

    def run(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        # 检查维度是否匹配
        # assert x_1.shape[0] == x_2.shape[0], "输入的两个向量必须有相同的特征数"

        kernel_value = rbf_jit(x_1, x_2)

        if x_2.ndim != 1 and x_2.shape[1] == 1:
            kernel_value = kernel_value.reshape(-1, 1)

        return kernel_value

    return run


def poly(p: int) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

    @jax.jit
    def run(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        # assert x_1.shape[0] == x_2.shape[0], "输入的两个向量必须有相同的特征数"

        return (x_1.T @ x_2) ** p

    return run


def linear() -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

    @jax.jit
    def run(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        # assert x_1.shape[0] == x_2.shape[0], "输入的两个向量必须有相同的特征数"

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
    ) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
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
        self._a_y_x: jnp.ndarray = None
        self._b: float = None
        self._kernel = Kernel.get_kernel(kernel_name, kernel_arg)
        self._kernel_info = {"name": kernel_name, "arg": kernel_arg}
        return

    @property
    def alpha(self) -> jnp.ndarray:
        return self._a_y_x[:, 0].reshape(-1, 1)

    @property
    def bias(self) -> jnp.ndarray:
        return self._b

    # TODO: use vmap to speed up
    def _build_k_matrix(self, x: jnp.ndarray) -> jnp.ndarray:
        size = x.shape[0]
        return jnp.array(
            [[self._kernel(x[i], x[j]) for j in range(size)] for i in range(size)]
        )

    def _find_alpha(
        self, K: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        size = x.shape[0]

        # K, x, y = np.asarray(K), np.asarray(x), np.asarray(y)

        alpha = cp.Variable(size)

        big_k = jnp.outer(y, y) * K
        big_k = cp.psd_wrap(big_k)

        objective = cp.Minimize((1 / 2) * cp.quad_form(alpha, big_k) - cp.sum(alpha))
        constraints = [cp.sum(cp.multiply(alpha, y)) == 0, alpha >= 0, alpha <= self._c]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.CLARABEL)  # verbose=True
        alpha = jnp.array(alpha.value)

        # filter the important one
        support_vectors = jnp.where((self._c >= alpha) & (alpha > self._threshold))[0]
        return alpha, support_vectors

    @staticmethod
    @jax.jit
    def _find_bias(
        alpha: jnp.ndarray,
        support_vectors: jnp.ndarray,
        y: jnp.ndarray,
        K: jnp.ndarray,
    ):
        K_hat = K[support_vectors][:, support_vectors]
        alpha_hat = alpha[support_vectors]
        y_hat = y[support_vectors]

        res = y_hat - K_hat @ (alpha_hat * y_hat)

        return jnp.mean(res)

    def train(
        self, x: Float[Array, "batch feature"], y: Float[Array, "batch 1"]
    ):  # [batch, feature]   [batch, 1]
        # get the a
        K = self._build_k_matrix(x)
        # print(K)
        alpha, support_vector = self._find_alpha(K, x, y)
        # find the best b
        self._b = SupportVectorMachine._find_bias(alpha, support_vector, y, K)

        alpha, y = alpha.reshape(-1, 1), y.reshape(-1, 1)

        table = jnp.hstack((alpha, y, x))

        self._a_y_x = table[support_vector]

        return

    def cal_one_item(
        self, ay: jnp.ndarray, x_kernel: jnp.ndarray, x_item: jnp.ndarray
    ) -> jnp.ndarray:
        # x [1, feature]
        # x_kernel [a, feature]

        @jax.jit
        def cal_one_item_jit(ay: jnp.ndarray, pre_x_kernel: jnp.ndarray, b: float):
            return jnp.sum(ay * pre_x_kernel) + b

        pre_x_kernel = self._kernel(x_kernel.T, x_item.reshape(-1, 1))

        return cal_one_item_jit(ay, pre_x_kernel, self._b)

    def __call__(self, x: jnp.ndarray, with_sign: bool = False) -> jnp.ndarray:
        # x [batch, feature]
        a, y, x_kernel = (
            self._a_y_x[:, 0].reshape(-1, 1),
            self._a_y_x[:, 1].reshape(-1, 1),
            self._a_y_x[:, 2:],
        )

        result = [
            self.cal_one_item(a * y, x_kernel=x_kernel, x_item=x_item) for x_item in x
        ]
        res = jnp.hstack(result)

        if with_sign:
            res = jnp.sign(res)

        return res

    def acc(self, x: jnp.ndarray, y: jnp.ndarray) -> tuple[float, jnp.ndarray]:

        y_hat = self(x, True)

        return jnp.mean(y_hat == y), y_hat

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
