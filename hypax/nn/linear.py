# parser.add_argument('--n-linear', type=int, default=2)
# hyperbolic_linear.py
#
# JAX/nnx implementation of a Poincaré fully-connected layer that mirrors
# `hypll.layers.HLinear` (PyTorch) while following the style of nnx.Linear.

from __future__ import annotations
import typing as tp

import jax.numpy as jnp
from flax import nnx

from hypax.array import ManifoldArray
from hypax.manifolds import Manifold


class HLinear(nnx.Module):
    """Hyperbolic (Poincaré) fully-connected layer for JAX/nnx."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: Manifold,
        *,
        use_bias: bool = True,
        dtype: tp.Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        """
        Args
        ----
        in_features / out_features : Input & output feature sizes.
        manifold                  : Instance of `hypax.manifolds.Manifold`.
        use_bias                  : Attach bias term (default **True**).
        dtype                     : Computation dtype (default: infer).
        param_dtype               : Parameter dtype (default **float32**).
        rngs                      : `nnx.Rngs` container (use `.params()`).
        """
        super().__init__()

        self.in_features = nnx.static(in_features)
        z_key = rngs.params()
        b_key = rngs.params() if use_bias else None
        weights, bias_value = manifold.construct_dl_parameters(
            in_features=in_features,
            out_features=out_features,
            bias=use_bias,
            key_z=z_key,
            key_bias=b_key,
            dtype=param_dtype,
        )

        self.weights = nnx.Param(jnp.asarray(weights, dtype=param_dtype))
        self.bias = (
            nnx.Param(jnp.asarray(bias_value, dtype=param_dtype))
            if bias_value is not None
            else None
        )

    def __call__(self, x: ManifoldArray) -> ManifoldArray:
        """Apply the hyperbolic fully connected operation."""
        if not isinstance(x, ManifoldArray):
            raise TypeError(f"Input must be a ManifoldArray, got {type(x)}")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )

        bias_value = self.bias.value if (self.bias is not None) else None
        result = x.manifold.fully_connected(
            x=x.data,
            z=self.weights.value,
            bias=bias_value,
            axis=-1,
        )
        return x.replace(data=result)
