import jax.numpy as jnp
import jax
from jax import vmap


def hl_gauss_transform(support, sigma=1):
    def transform_to_probs(target: jax.Array) -> jax.Array:
        cdf_evals = jax.scipy.special.erf((support - target) / (jnp.sqrt(2) * sigma))
        z = cdf_evals[-1] - cdf_evals[0]
        bin_probs = cdf_evals[1:] - cdf_evals[:-1]
        return bin_probs / z

    def transform_from_probs(probs: jax.Array) -> jax.Array:
        centers = (support[:-1] + support[1:]) / 2
        return jnp.sum(probs * centers)

    return vmap(transform_to_probs, in_axes=0), vmap(transform_from_probs, in_axes=0)


def hl_gauss_targets(rewards, discounts, target_prob_distribution, support):
    target = rewards + discounts * target_prob_distribution @ support
    to_probs, _ = hl_gauss_transform(support)
    target_prob = to_probs(target)
    return target_prob


def project_distribution(projected_from, weights, projected_to):
    v_min, v_max = projected_to[0], projected_to[-1]
    B, N = projected_from.shape
    delta_z = (v_max - v_min) / (N - 1)
    b = (projected_from - v_min) / delta_z
    l = jnp.floor(b).astype(jnp.int32)
    u = l + 1

    l_index = jnp.clip(l, 0, N - 1)
    u_index = jnp.clip(u, 0, N - 1)
    batch_idx = jnp.arange(B)[:, None]

    m = jnp.zeros_like(projected_from)
    m = m.at[batch_idx, l_index].add(weights * (u - b))
    m = m.at[batch_idx, u_index].add(weights * (b - l))

    return m

def c51_targets(rewards, discounts, target_prob_distribution, support):
    v_min, v_max = support[0], support[-1]
    target_support = rewards[:, None] + jnp.outer(discounts, support)
    target_support = jnp.clip(target_support, v_min, v_max)
    print(target_support.shape, target_prob_distribution.shape, support.shape)
    targets = project_distribution(target_support, target_prob_distribution, support)
    return targets
