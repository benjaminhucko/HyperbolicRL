# convolution.py
#
# JAX/nnx implementation of a Poincaré 2D convolution layer

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
from flax import nnx
from hypax.array import ManifoldArray
from hypax.manifolds import Manifold
from hypax.manifolds.poincare_ball._linalg import (
    poincare_unfold,
    poincare_fully_connected,
)
from hypax.nn._layer_init import construct_conv_parameters


def _normalize_kernel_size(kernel_size: int | Tuple[int, int]) -> Tuple[int, int]:
    """Normalize kernel_size to a tuple of (height, width)."""
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    return kernel_size


class HConvolution2D(nnx.Module):
    """Hyperbolic 2D convolution layer for the Poincaré ball model.

    This layer implements convolution in hyperbolic space using the approach from
    HNN++ (Chen et al., 2022). The operation consists of three steps:
    1. Unfold: Extract patches with beta-concatenation rescaling
    2. Fully connected: Apply hyperbolic linear transformation
    3. Reshape: Reform the spatial structure

    Example:
        >>> from hypax.manifolds.poincare_ball import PoincareBall
        >>> manifold = PoincareBall(c=1.0)
        >>> conv = HConvolution2D(
        ...     in_channels=3,
        ...     out_channels=16,
        ...     kernel_size=5,
        ...     manifold=manifold,
        ...     rngs=nnx.Rngs(0)
        ... )
        >>> x = ManifoldArray(
        ...     data=jnp.zeros((1, 3, 32, 32)),
        ...     manifold=manifold
        ... )
        >>> y = conv(x)
        >>> y.shape
        (1, 16, 28, 28)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        manifold: Manifold,
        *,
        bias: bool = True,
        stride: int = 1,
        padding: int = 0,
        id_init: bool = True,
        dtype: Optional[jnp.dtype] = None,
        param_dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        """
        Args
        ----
        in_channels  : Number of input channels.
        out_channels : Number of output channels.
        kernel_size  : Size of the convolving kernel (int or tuple).
        manifold     : Instance of `Manifold` (should have curvature attribute).
        bias         : Add learnable bias (default **True**).
        stride       : Stride of the convolution (default **1**).
        padding      : Zero-padding added to both sides of input (default **0**).
        id_init      : Use identity initialization vs. HNN++ init (default **True**).
        dtype        : Computation dtype (default: infer).
        param_dtype  : Parameter dtype (default **float32**).
        rngs         : `nnx.Rngs` container (use `.params()`).
        """
        super().__init__()

        # Validate manifold has curvature attribute
        if not hasattr(manifold, "curvature"):
            raise ValueError(
                "Manifold must have a curvature attribute for hyperbolic convolution"
            )

        # Normalize kernel_size to tuple
        self.kernel_size = nnx.static(_normalize_kernel_size(kernel_size))
        kernel_h, kernel_w = self.kernel_size
        self.kernel_vol = nnx.static(kernel_h * kernel_w)

        # Store configuration
        self.in_channels = nnx.static(in_channels)
        self.out_channels = nnx.static(out_channels)
        self.has_bias = nnx.static(bias)
        self.stride = nnx.static(stride)
        self.padding = nnx.static(padding)
        self.id_init = nnx.static(id_init)
        self.dtype = nnx.static(dtype)
        self.param_dtype = nnx.static(param_dtype)

        # Initialize parameters
        param_key = rngs.params()
        weights, bias_init = construct_conv_parameters(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            use_bias=bias,
            key=param_key,
            dtype=param_dtype,
            id_init=id_init,
        )
        # Store as nnx parameters
        self.weights = nnx.Param(
            weights
        )  # shape: [kernel_vol * in_channels, out_channels]
        self.bias = (
            nnx.Param(bias_init) if bias else None
        )  # shape: [out_channels] or None

    def __call__(self, x: ManifoldArray) -> ManifoldArray:
        """Apply the hyperbolic 2D convolution.

        Args:
            x: Input ManifoldArray with shape [batch, in_channels, height, width]

        Returns:
            ManifoldArray with shape [batch, out_channels, out_height, out_width]
        """
        # Validate input
        if not isinstance(x, ManifoldArray):
            raise TypeError(f"Input must be a ManifoldArray, got {type(x)}")

        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {channels}"
            )

        # Extract curvature from manifold
        c = x.manifold.curvature()

        # Calculate output spatial dimensions
        kernel_h, kernel_w = self.kernel_size
        out_height = (height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_w) // self.stride + 1

        # Step 1: Hyperbolic unfold with beta-concatenation
        # Input: [batch, in_channels, height, width]
        # Output: [batch, kernel_vol * in_channels, num_patches]
        x_unfolded = poincare_unfold(
            x=x.data,
            kernel_size=self.kernel_size,
            in_channels=self.in_channels,
            c=c,
            stride=self.stride,
            padding=self.padding,
            axis=1,  # Channel axis
        )

        # Step 2: Apply hyperbolic fully connected layer
        # Input: [batch, kernel_vol * in_channels, num_patches]
        # Weights: [kernel_vol * in_channels, out_channels]
        # Output: [batch, out_channels, num_patches]
        bias_value = self.bias.value if self.has_bias else None
        x_fc = poincare_fully_connected(
            x=x_unfolded,
            z=self.weights.value,
            bias=bias_value,
            c=c,
            axis=1,  # Apply FC along the feature dimension
        )

        # Step 3: Reshape to spatial format
        # Input: [batch, out_channels, num_patches]
        # Output: [batch, out_channels, out_height, out_width]
        x_reshaped = x_fc.reshape(batch_size, self.out_channels, out_height, out_width)
        # Return as ManifoldArray
        return x.replace(data=x_reshaped)
