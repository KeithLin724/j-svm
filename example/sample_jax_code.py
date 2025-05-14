import jax
import time
import jax.numpy as jnp

a = jnp.array([[1, 2, 3], [4, 5, 6]])
b = jnp.array([[7, 8], [9, 10], [11, 12]])


@jax.jit
def dummy_func(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
    # assert x_1.shape[0] == x_2.shape[0], "输入的两个向量必须有相同的特征数"
    return x_1 @ x_2 * 2


def dummy_func2(x_1: jnp.ndarray, x_2: jnp.ndarray) -> jnp.ndarray:
    # assert x_1.shape[0] == x_2.shape[0], "输入的两个向量必须有相同的特征数"
    return x_1 @ x_2 * 2


print(a @ b)

# Warm up JIT
dummy_func(a, b).block_until_ready()

a.to_device(jax.devices()[0])
b.to_device(jax.devices()[0])

# Time dummy_func (JIT)
start = time.time()
for _ in range(10000):
    res1 = dummy_func(a, b)
jax.block_until_ready(res1)
end = time.time()
print("dummy_func (JIT) time:", end - start)

# Time dummy_func2 (no JIT)
start = time.time()
for _ in range(10000):
    res2 = dummy_func2(a, b)
jax.block_until_ready(res2)
end = time.time()
print("dummy_func2 (no JIT) time:", end - start)

# import jax
# import jax.numpy as jnp

# print(jax.devices())  # 列出所有可用裝置（含 GPU）


# # 定義要平行執行的函式
# def parallel_matmul(a, b):
#     return jnp.dot(a, b)


# # 使用 pmap 同步映射到多個 GPU
# parallel_func = jax.pmap(parallel_matmul, axis_name="devices")

# # 假設有兩張 GPU，就需在第 0 軸擺放對應次數的批次資料
# # 以下為範例，實際需依照資料大小切分
# a = jnp.stack([jnp.ones((2, 2)), jnp.ones((2, 2))])
# b = jnp.stack([jnp.ones((2, 2)), jnp.ones((2, 2))])

# # 把資料同時送往各 GPU 執行
# result = parallel_func(a, b)
# print(result)
