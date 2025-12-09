import jax
import jax.numpy as jnp

from optax import GradientTransformation


def riemannian_adam(
    lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=False
) -> GradientTransformation:
    beta1, beta2 = betas

    def _zeros_like_accumulator(ref):
        if ref.ndim == 0:
            return jnp.zeros_like(ref)
        shape = ref.shape[:-1] + (1,)
        return jnp.zeros(shape, dtype=ref.dtype)

    def _sum_squares_last_axis(arr):
        if arr.ndim == 0:
            return arr**2
        return jnp.sum(arr**2, axis=-1, keepdims=True)

    def init_fn(params):
        def _init_leaf(p):
            m = jnp.zeros_like(p["tensor"] if isinstance(p, dict) else p)
            v = _zeros_like_accumulator(m)
            buf = {"m": m, "v": v, "step": jnp.zeros([], jnp.int32)}
            if amsgrad:
                buf["v_hat"] = jnp.zeros_like(v)
            return buf

        return jax.tree.map(_init_leaf, params)

    def update_fn(grads, state, params):
        def _update_leaf(p, g, s):
            step = s["step"] + 1

            # manifold leaf
            if isinstance(p, dict):
                M, x = p["manifold"], p["tensor"]
                g = g + weight_decay * x
                g = M.euc_to_tangent(x, g)
                m = beta1 * s["m"] + (1 - beta1) * g
                v = beta2 * s["v"] + (1 - beta2) * M.inner(
                    x,
                    g,
                    g,
                    keepdims=True,
                )

                v_corr = v / (1 - beta2**step)
                if amsgrad:
                    v_hat = jnp.maximum(s["v_hat"], v)
                    v_corr = v_hat / (1 - beta2**step)

                m_corr = m / (1 - beta1**step)
                update = -lr * m_corr / (jnp.sqrt(v_corr) + eps)  # tangent
                new_state = {"m": m, "v": v, "step": step}
                if amsgrad:
                    new_state["v_hat"] = v_hat
                return update, new_state

            else:
                # euclidean leaf
                g = g + weight_decay * p
                m = beta1 * s["m"] + (1 - beta1) * g
                v = beta2 * s["v"] + (1 - beta2) * _sum_squares_last_axis(g)

                v_corr = v / (1 - beta2**step)
                if amsgrad:
                    v_hat = jnp.maximum(s["v_hat"], v)
                    v_corr = v_hat / (1 - beta2**step)

                m_corr = m / (1 - beta1**step)
                update = -lr * m_corr / (jnp.sqrt(v_corr) + eps)
                new_state = {"m": m, "v": v, "step": step}
                if amsgrad:
                    new_state["v_hat"] = v_hat
                return update, new_state

        leaf_results = jax.tree_util.tree_map(
            _update_leaf,
            params,
            grads,
            state,
            is_leaf=lambda x: isinstance(x, tuple),  # keep tuples closed
        )

        updates = jax.tree_util.tree_map(
            lambda t: t[0], leaf_results, is_leaf=lambda x: isinstance(x, tuple)
        )
        new_state = jax.tree_util.tree_map(
            lambda t: t[1], leaf_results, is_leaf=lambda x: isinstance(x, tuple)
        )

        return updates, new_state

    return GradientTransformation(init_fn, update_fn)
