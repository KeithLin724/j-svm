"""Benchmark JAX JIT vs no-JIT performance for matrix multiplication."""
# %% imports
import jax
import time
import jax.numpy as jnp

# %% setup
if __name__ == "__main__":
    a = jnp.array([[1, 2, 3], [4, 5, 6]])
    b = jnp.array([[7, 8], [9, 10], [11, 12]])


    @jax.jit
    def dummy_func(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        return x_1 @ x_2 * 2


    def dummy_func2(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
        return x_1 @ x_2 * 2


    print(a @ b)
    dummy_func(a, b).block_until_ready()

    a.to_device(jax.devices()[0])
    b.to_device(jax.devices()[0])

# %% benchmark
if __name__ == "__main__":
    start = time.time()
    for _ in range(10000):
        res1 = dummy_func(a, b)
    jax.block_until_ready(res1)
    end = time.time()
    print("dummy_func (JIT) time:", end - start)

    start = time.time()
    for _ in range(10000):
        res2 = dummy_func2(a, b)
    jax.block_until_ready(res2)
    end = time.time()
    print("dummy_func2 (no JIT) time:", end - start)
