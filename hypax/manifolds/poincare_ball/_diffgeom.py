import jax
import jax.numpy as jnp

from hypax.manifolds.poincare_ball._stats import safe_norm


def mobius_add(x: jax.Array, y: jax.Array, c: jax.Array, axis: int = -1) -> jax.Array:
    broadcast_dim = max(x.ndim, y.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis

    x2 = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
    y2 = jnp.square(y).sum(axis=axis - broadcast_dim + y.ndim, keepdims=True)

    xy = (x * y).sum(axis=axis, keepdims=True)

    numerator = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denominator = jnp.maximum(1 + 2 * c * xy + c**2 * x2 * y2, 1e-15)

    return numerator / denominator


def project(x: jax.Array, c: jax.Array, axis: int = -1, eps: float = -1.0) -> jax.Array:
    eps = jnp.asarray(eps, dtype=x.dtype)
    eps_val = jnp.where(
        eps < 0, jnp.array(4e-3 if x.dtype == jnp.float32 else 1e-5, dtype=x.dtype), eps
    )

    maxnorm = (1 - eps_val) / jnp.sqrt(c + 1e-15)
    maxnorm = jnp.where(c > 0, maxnorm, jnp.full_like(c, 1e15))
    norm = safe_norm(x, axis)

    cond = norm > maxnorm
    projected = x / norm * maxnorm

    return jnp.where(cond, projected, x)


def expmap0(v: jax.Array, c: jax.Array, axis: int = -1):
    v_norm = safe_norm(v, axis)
    v_norm_c_sqrt = v_norm * jnp.sqrt(c)
    return project(jnp.tanh(v_norm_c_sqrt) * v / v_norm_c_sqrt, c, axis=axis)


def logmap0(y: jax.Array, c: jax.Array, axis: int = -1):
    y_norm = safe_norm(y, axis)
    y_norm_c_sqrt = y_norm * jnp.sqrt(c)
    return jnp.atanh(y_norm_c_sqrt) * y / y_norm_c_sqrt


def expmap(x: jax.Array, v: jax.Array, c: jax.Array, axis: int = -1):
    broadcast_dim = max(x.ndim, v.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis

    v_norm = safe_norm(v, axis=axis - broadcast_dim + v.ndim)

    x_norm_sq = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
    lambda_x = 2 / jnp.clip(1 - c * x_norm_sq, min=1e-15)

    c_sqrt = jnp.sqrt(c)
    second_term = jnp.tanh(c_sqrt * lambda_x * v_norm / 2) * v / (c_sqrt * v_norm)

    return project(mobius_add(x, second_term, c, axis=axis), c, axis=axis)


def logmap(x: jax.Array, y: jax.Array, c: jax.Array, axis: int = -1):
    broadcast_dim = max(x.ndim, y.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis

    min_x_y = mobius_add(-x, y, c, axis=axis)
    min_x_y_norm = safe_norm(min_x_y, axis=axis)

    x_norm_sq = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
    lambda_x = 2 / jnp.clip(1 - c * x_norm_sq, min=1e-15)

    c_sqrt = jnp.sqrt(c)
    return (
        2
        / (c_sqrt * lambda_x)
        * jnp.atanh(c_sqrt * min_x_y_norm)
        * min_x_y
        / min_x_y_norm
    )


def gyration(u: jax.Array, v: jax.Array, w: jax.Array, c: jax.Array, axis: int = -1):
    broadcast_dim = max(u.ndim, v.ndim, w.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis
    u2 = jnp.sum(u**2, axis=axis - broadcast_dim + u.ndim, keepdims=True)
    v2 = jnp.sum(v**2, axis=axis - broadcast_dim + v.ndim, keepdims=True)
    uv = jnp.sum(u * v, axis=axis - broadcast_dim + max(u.ndim, v.ndim), keepdims=True)
    uw = jnp.sum(u * w, axis=axis - broadcast_dim + max(u.ndim, w.ndim), keepdims=True)
    vw = jnp.sum(v * w, axis=axis - broadcast_dim + max(v.ndim, w.ndim), keepdims=True)

    K2 = c**2
    a = -K2 * uw * v2 + c * vw + 2 * K2 * uv * vw
    b = -K2 * vw * u2 - c * uw
    d = 1 + 2 * c * uv + K2 * u2 * v2

    return w + 2 * (a * u + b * v) / jnp.maximum(d, 1e-15)


def transp(x: jax.Array, y: jax.Array, v: jax.Array, c: jax.Array, axis: int = -1):
    broadcast_dim = max(x.ndim, y.ndim, v.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis

    x_norm_sq = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
    lambda_x = 2 / jnp.clip(1 - c * x_norm_sq, min=1e-15)

    y_norm_sq = jnp.square(y).sum(axis=axis - broadcast_dim + y.ndim, keepdims=True)
    lambda_y = 2 / jnp.clip(1 - c * y_norm_sq, min=1e-15)

    return gyration(y, -x, v, c, axis=axis) * lambda_x / lambda_y


def dist(
    x: jax.Array, y: jax.Array, c: jax.Array, axis: int = -1, keepdims: bool = False
) -> jax.Array:
    return (
        2
        / jnp.sqrt(c)
        * jnp.atanh(
            (
                jnp.sqrt(c)
                * jnp.linalg.norm(
                    mobius_add(-x, y, c, axis=axis), axis=axis, keepdims=keepdims
                )
            )
        )
    )


def inner(
    x: jax.Array,
    u: jax.Array,
    v: jax.Array,
    c: jax.Array,
    axis: int = -1,
    keepdims: bool = False,
) -> jax.Array:
    broadcast_dim = max(x.ndim, u.ndim, v.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis
    x_norm_sq = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
    lambda_x = 2 / jnp.clip(1 - c * x_norm_sq, min=1e-15)
    dot_prod = (u * v).sum(axis=axis, keepdims=keepdims)
    return jnp.square(lambda_x) * dot_prod


def euc_to_tangent(
    x: jax.Array, u: jax.Array, c: jax.Array, axis: int = -1
) -> jax.Array:
    broadcast_dim = max(x.ndim, u.ndim)
    axis = axis if axis >= 0 else broadcast_dim + axis
    x_norm_sq = jnp.square(x).sum(axis=axis - broadcast_dim + x.ndim, keepdims=True)
    lambda_x = 2 / jnp.clip(1 - c * x_norm_sq, min=1e-15)
    return u / jnp.square(lambda_x)


def mobius_add_batch(x: jax.Array, y: jax.Array, c: jax.Array) -> jax.Array:
    xy = jnp.einsum("bij,bkj->bik", x, y)
    x2 = jnp.square(x).sum(axis=-1, keepdims=True)
    y2 = jnp.square(y).sum(axis=-1, keepdims=True)
    num = 1 + 2 * c * xy + c * jnp.permute_dims(y2, (0, 2, 1))
    num = jnp.expand_dims(num, axis=2) * jnp.expand_dims(x, axis=2)
    num = num + jnp.expand_dims(1 - c * x2, axis=3) * jnp.expand_dims(y, axis=1)
    denom = 1 + 2 * c * xy + jnp.square(c) * x2 * jnp.permute_dims(y2, (0, 2, 1))
    return num / jnp.clip(jnp.expand_dims(denom, axis=3), min=1e-15)


def _ensure_batch_dim(tensor: jax.Array) -> tuple[jax.Array, bool]:
    if tensor.ndim == 3:
        return tensor, False
    if tensor.ndim == 2:
        return tensor[jnp.newaxis, ...], True
    raise ValueError("Expected tensor with 2 or 3 dimensions")


def cdist(x: jax.Array, y: jax.Array, c: jax.Array) -> jax.Array:
    x, squeezed_x = _ensure_batch_dim(x)
    y, squeezed_y = _ensure_batch_dim(y)

    if x.shape[0] != y.shape[0]:
        if x.shape[0] == 1:
            x = jnp.broadcast_to(x, (y.shape[0],) + x.shape[1:])
            squeezed_x = False
        elif y.shape[0] == 1:
            y = jnp.broadcast_to(y, (x.shape[0],) + y.shape[1:])
            squeezed_y = False
        else:
            raise ValueError(
                f"Cannot broadcast batch dimensions: x has {x.shape[0]}, y has {y.shape[0]}"
            )

    diffs = mobius_add_batch(-x, y, c)
    norm = jnp.linalg.norm(diffs, axis=-1)
    distances = 2 / jnp.sqrt(c) * jnp.atanh(jnp.sqrt(c) * norm)

    if x.shape[0] == 1 and squeezed_x and squeezed_y:
        return distances[0]
    return distances
