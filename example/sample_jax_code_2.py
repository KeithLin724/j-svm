import numpy as np
import jax
import time
import jax.numpy as jnp


# Matrix dimensions
N = 1000
A_np = np.random.randn(N, N)
B_np = np.random.randn(N, N)
A_jnp = jnp.array(A_np)
B_jnp = jnp.array(B_np)


def np_matmul(a, b):
    return a @ b


@jax.jit
def jnp_matmul(a, b):
    return a @ b


def jnp_matmul_no_jit(a, b):
    return a @ b


# Warm up JAX JIT
_ = jnp_matmul(A_jnp, B_jnp).block_until_ready()

# Time numpy matmul
start = time.time()
np_result = np_matmul(A_np, B_np)
np_time = time.time() - start

# Time jax.jit matmul
start = time.time()
jnp_result = jnp_matmul(A_jnp, B_jnp)
jnp_time = time.time() - start

# Time jax.jit matmul
start = time.time()
jnp_result = jnp_matmul_no_jit(A_jnp, B_jnp)
jnp_time_no_jit = time.time() - start

print(f"NumPy matmul time: {np_time:.6f} seconds")
print(f"JAX JIT matmul time: {jnp_time:.6f} seconds")
print(f"JAX no JIT matmul time: {jnp_time_no_jit:.6f} seconds")

# Example: Using jax.vmap to batch matrix-vector multiplication

# Create a batch of vectors
batch_size = 10
vectors = jnp.array(np.random.randn(batch_size, N))
print(f"Batch of vectors shape: {vectors.shape}")


# Define a function for matrix-vector multiplication
def matvec(a, x):
    return a @ x


# Vectorize matvec over the batch dimension of vectors
batched_matvec = jax.vmap(matvec, in_axes=(None, 0))
batched_matvec_jit = jax.jit(batched_matvec)

# Warm up JIT
_ = batched_matvec_jit(A_jnp, vectors).block_until_ready()

# Time non-JIT version
start = time.time()
result_no_jit = batched_matvec(A_jnp, vectors)
jax.block_until_ready(result_no_jit)
time_no_jit = time.time() - start

# Time JIT version
start = time.time()
result_jit = batched_matvec_jit(A_jnp, vectors)
jax.block_until_ready(result_jit)
time_jit = time.time() - start

print(f"Batched matrix-vector multiplication result shape: {result_jit.shape}")
print(f"JAX vmap no JIT time: {time_no_jit:.6f} seconds")
print(f"JAX vmap with JIT time: {time_jit:.6f} seconds")
