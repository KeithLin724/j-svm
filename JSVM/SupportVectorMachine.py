import jax
import jax.numpy as jnp

# import numpy as np
import cvxpy as cp
from jaxtyping import Array, Float
from typing import Callable
from dataclasses import dataclass
import time

from core import SVMParameter
from pathlib import Path

# https://jaxopt.github.io/stable/quadratic_programming.html
from jaxopt import BoxOSQP
from functools import cache


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

    _ = rbf_jit(jnp.zeros((1,)), jnp.zeros((1,))).block_until_ready()  # warm up

    return run


def poly(p: int) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

    @jax.jit
    def run(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        # assert x_1.shape[0] == x_2.shape[0], "输入的两个向量必须有相同的特征数"

        return (x_1.T @ x_2) ** p

    _ = run(jnp.zeros((1,)), jnp.zeros((1,))).block_until_ready()  # warm up

    return run


def linear() -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:

    @jax.jit
    def run(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        # assert x_1.shape[0] == x_2.shape[0], "输入的两个向量必须有相同的特征数"

        return x_1.T @ x_2

    _ = run(jnp.zeros((1,)), jnp.zeros((1,))).block_until_ready()  # warm up

    return run


@dataclass(slots=True)
class KernelResult:
    kernel_function: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    fast_forward_jit: Callable
    build_k_matrix: Callable[[jnp.ndarray], jnp.ndarray]


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

    @cache
    @staticmethod
    def fast_build_jit_function(name: str, config: tuple):

        kernel = Kernel.get_kernel(name, dict(config))

        @jax.jit
        def build_k_matrix(x: Float[Array, "N D"]) -> jnp.ndarray:

            # 先對 x 的每一列做 vmap，然後再對每一列的每一個元素做 vmap
            # K[i, j] = kernel(x[i], x[j])
            return jax.vmap(lambda xi: jax.vmap(lambda xj: kernel(xi, xj))(x))(x)

        @jax.jit
        def cal_one_item(
            ay: jnp.ndarray, x_kernel: jnp.ndarray, x_item: jnp.ndarray, b: float
        ):
            return jnp.sum(ay * kernel(x_kernel.T, x_item.reshape(-1, 1))) + b

        @jax.jit
        def fast_forward_jit(
            a_y_x: jnp.ndarray, x: jnp.ndarray, b: float
        ) -> jnp.ndarray:

            a, y, x_kernel = (
                a_y_x[:, 0].reshape(-1, 1),
                a_y_x[:, 1].reshape(-1, 1),
                a_y_x[:, 2:],
            )

            process_x = jax.vmap(
                lambda x_item: cal_one_item(a * y, x_kernel, x_item, b)
            )

            result = process_x(x)
            res = jnp.hstack(result)

            return res

        ## warm up

        _ = build_k_matrix(jnp.zeros((1, 1))).block_until_ready()

        _ = cal_one_item(
            jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.zeros((1, 1)), 0
        ).block_until_ready()

        _ = fast_forward_jit(
            jnp.zeros((1, 1)), jnp.zeros((1, 1)), 0
        ).block_until_ready()

        return KernelResult(
            kernel_function=kernel,
            fast_forward_jit=fast_forward_jit,
            build_k_matrix=build_k_matrix,
        )

    @staticmethod
    def build_fast_jit_function_v2(name: str, config: dict, with_time: bool = None):
        # 這裡 config 是一個字典，所以要轉換成 tuple

        return Kernel.fast_build_jit_function(name, tuple(config.items()))

    @staticmethod
    def build_fast_jit_function(name: str, config: dict, with_time: bool = False):
        kernel = Kernel.get_kernel(name, config)

        @jax.jit
        def build_k_matrix(x: Float[Array, "N D"]) -> jnp.ndarray:

            # 先對 x 的每一列做 vmap，然後再對每一列的每一個元素做 vmap
            # K[i, j] = kernel(x[i], x[j])
            return jax.vmap(lambda xi: jax.vmap(lambda xj: kernel(xi, xj))(x))(x)

        @jax.jit
        def cal_one_item(
            ay: jnp.ndarray, x_kernel: jnp.ndarray, x_item: jnp.ndarray, b: float
        ):
            return jnp.sum(ay * kernel(x_kernel.T, x_item.reshape(-1, 1))) + b

        @jax.jit
        def fast_forward_jit(
            a_y_x: jnp.ndarray, x: jnp.ndarray, b: float
        ) -> jnp.ndarray:

            a, y, x_kernel = (
                a_y_x[:, 0].reshape(-1, 1),
                a_y_x[:, 1].reshape(-1, 1),
                a_y_x[:, 2:],
            )

            process_x = jax.vmap(
                lambda x_item: cal_one_item(a * y, x_kernel, x_item, b)
            )

            result = process_x(x)
            res = jnp.hstack(result)

            return res

        ## warm up
        if with_time:

            start = time.time()
            _ = build_k_matrix(jnp.zeros((1, 1))).block_until_ready()
            end = time.time()
            print(f"build_k_matrix time: {end - start:.4f} s")

            start = time.time()
            _ = cal_one_item(
                jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.zeros((1, 1)), 0
            ).block_until_ready()
            end = time.time()
            print(f"cal_one_item time: {end - start:.4f} s")

            start = time.time()
            _ = fast_forward_jit(
                jnp.zeros((1, 1)), jnp.zeros((1, 1)), 0
            ).block_until_ready()
            end = time.time()
            print(f"fast_forward_jit time: {end - start:.4f} s")
        else:
            _ = build_k_matrix(jnp.zeros((1, 1))).block_until_ready()

            _ = cal_one_item(
                jnp.zeros((1, 1)), jnp.zeros((1, 1)), jnp.zeros((1, 1)), 0
            ).block_until_ready()

            _ = fast_forward_jit(
                jnp.zeros((1, 1)), jnp.zeros((1, 1)), 0
            ).block_until_ready()

        return KernelResult(
            kernel_function=kernel,
            fast_forward_jit=fast_forward_jit,
            build_k_matrix=build_k_matrix,
        )


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
        # self._kernel = Kernel.get_kernel(kernel_name, kernel_arg)
        self._kernel_info = {"name": kernel_name, "arg": kernel_arg}

        self.__kernel_result = Kernel.build_fast_jit_function_v2(
            kernel_name, kernel_arg, with_time=True
        )

        self._build_k_matrix = self.__kernel_result.build_k_matrix
        self._kernel = self.__kernel_result.kernel_function
        self._fast_forward_jit = self.__kernel_result.fast_forward_jit

        return

    def save(self, path: str | Path) -> None:

        parameter = SVMParameter(
            C=self._c,
            threshold=self._threshold,
            kernel_name=self._kernel_info["name"],
            kernel_arg=self._kernel_info["arg"],
            a_y_x=self._a_y_x,
            b=self._b,
        )

        parameter.save(path)

        return

    @staticmethod
    def load_from(path: str | Path):

        parameter = SVMParameter.load_from(path)

        model = SupportVectorMachine(
            C=parameter.C,
            kernel_name=parameter.kernel_name,
            kernel_arg=parameter.kernel_arg,
            threshold=parameter.threshold,
        )

        model._a_y_x = parameter.a_y_x
        model._b = parameter.b

        return model

    @staticmethod
    def warm_up():
        ## jit warm up

        key = jax.random.PRNGKey(0)
        alpha = jax.random.uniform(key, (5,))
        support_vectors = jnp.array([0, 2, 4])
        y = jax.random.choice(key, jnp.array([-1.0, 1.0]), (5,))
        K = jax.random.normal(key, (5, 5))

        _ = SupportVectorMachine.__find_bias(
            alpha, support_vectors, y, K
        ).block_until_ready()

        _ = SupportVectorMachine.__train_jit(
            alpha, support_vectors, y, K
        ).block_until_ready()

        _ = SupportVectorMachine.__find_alpha_jit(K, y, 1.0).block_until_ready()
        return

    @property
    def alpha(self) -> jnp.ndarray:
        return self._a_y_x[:, 0].reshape(-1, 1)

    @property
    def bias(self) -> jnp.ndarray:
        return self._b

    def _find_alpha(
        self, K: jnp.ndarray, y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:

        alpha = SupportVectorMachine.__find_alpha_jit(K, y, self._c)

        # filter the important one
        support_vectors = jnp.where((self._c >= alpha) & (alpha > self._threshold))[0]

        return alpha, support_vectors

    @staticmethod
    def __matvec_A(_, beta):
        return beta, jnp.sum(beta)

    @staticmethod
    @jax.jit
    def __find_alpha_jit(
        K: jnp.ndarray,
        y: jnp.ndarray,
        C: float,
        # threshold: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        C, y = jnp.float32(C), y.astype(jnp.float32)

        # l, u must have same shape than matvec_A's output.
        l = -jax.nn.relu(-y * C), 0.0
        u = jax.nn.relu(y * C), 0.0

        # big_k = jnp.outer(y, y) * K
        osqp = BoxOSQP(matvec_A=SupportVectorMachine.__matvec_A)

        sol, _ = osqp.run(params_obj=(K, -y), params_eq=None, params_ineq=(l, u))

        alpha = sol.primal[0]

        return alpha

    @staticmethod
    @jax.jit
    def __find_bias(
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

    @staticmethod
    @jax.jit
    def __train_jit(
        alpha: jnp.ndarray,
        support_vector: jnp.ndarray,
        y: jnp.ndarray,
        x: jnp.ndarray,
    ):

        alpha, y = alpha.reshape(-1, 1), y.reshape(-1, 1)

        table = jnp.hstack((alpha, y, x))

        a_y_x = table[support_vector]
        return a_y_x

    def train(
        self, x: Float[Array, "batch feature"], y: Float[Array, "batch 1"]
    ):  # [batch, feature]   [batch, 1]
        # get the a
        x, y = jnp.asarray(x), jnp.asarray(y)
        K = self._build_k_matrix(x)

        alpha, support_vector = self._find_alpha(K, y)
        # find the best b

        self._b = SupportVectorMachine.__find_bias(alpha, support_vector, y, K)

        self._a_y_x = SupportVectorMachine.__train_jit(alpha, support_vector, y, x)

        return

    def __call__(self, x: jnp.ndarray, with_sign: bool = False) -> jnp.ndarray:
        # x [batch, feature]
        res = self._fast_forward_jit(self._a_y_x, x, self._b)

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
