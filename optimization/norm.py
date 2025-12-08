import jax
import jax.numpy as jnp

def spectral_norm(weights, n_power_iterations=1, eps=1e-12):
    def power_method(W):
        key = jax.random.PRNGKey(0)
        v = jax.random.normal(key, (W.shape[1],))
        for _ in range(n_power_iterations):
            u = W @ v
            u = u / (jnp.linalg.norm(u) + eps)
            v = W.T @ u
            v = v / (jnp.linalg.norm(v) + eps)
        return u, v

    u, v = power_method(weights)
    sigma = u @ (weights @ v)
    return weights / sigma

def normalize(x, epsilon=1e-8):
    normalized = (x - jnp.mean(x)) / (jnp.std(x) + epsilon)
    return normalized