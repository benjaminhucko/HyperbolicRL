# _init.py
#
# Parameter initialization utilities for hyperbolic neural network layers

from __future__ import annotations
from typing import Tuple, Optional

import jax
import jax.numpy as jnp


def construct_conv_parameters(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    use_bias: bool,
    key: jax.Array,
    dtype: jnp.dtype = jnp.float32,
    id_init: bool = True,
) -> Tuple[jax.Array, Optional[jax.Array]]:
    """Construct and initialize parameters for hyperbolic 2D convolution.

    Follows the initialization strategy from HNN++:
    - Identity initialization: For square or tall weight matrices (in_features <= out_features),
      initialize weights to (1/2) * eye(in_features, out_features)
    - HNN++ initialization: For wide matrices (in_features > out_features),
      initialize with normal distribution: N(0, (2 * in_features * out_features)^-0.5)
    - Bias is always initialized to zeros

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel as (kernel_h, kernel_w)
        use_bias: Whether to create bias parameters
        key: JAX random key for initialization
        dtype: Data type for parameters (default: float32)
        id_init: Use identity initialization (True) or HNN++ initialization (False)

    Returns:
        Tuple of (weights, bias) where:
        - weights: Array of shape [kernel_vol * in_channels, out_channels]
        - bias: Array of shape [out_channels] if use_bias=True, else None

    References:
        Chen et al. "Fully Hyperbolic Neural Networks" (HNN++), ACL 2022
    """
    kernel_h, kernel_w = kernel_size
    kernel_vol = kernel_h * kernel_w
    in_features = kernel_vol * in_channels
    out_features = out_channels

    # Split key for weights and bias
    key_w, key_b = jax.random.split(key)

    # Initialize weights based on id_init flag and matrix shape
    if id_init and in_features <= out_features:
        # Identity initialization for square or tall matrices
        weights = 0.5 * jnp.eye(in_features, out_features, dtype=dtype)
    else:
        # HNN++ initialization for wide matrices or when id_init=False
        std = (2 * in_features * out_features) ** -0.5
        weights = (
            jax.random.normal(key_w, shape=(in_features, out_features), dtype=dtype)
            * std
        )

    # Initialize bias to zeros if needed
    bias = jnp.zeros(out_features, dtype=dtype) if use_bias else None

    return weights, bias
