# activation.py
#
# JAX/nnx implementation of hyperbolic activation functions for the Poincaré ball model.

from __future__ import annotations
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from hypax.array import ManifoldArray
from hypax.manifolds.poincare_ball._diffgeom import logmap0, expmap0


def hrelu(
    x: ManifoldArray,
    c: Optional[jax.Array] = None,
    axis: int = -1,
) -> ManifoldArray:
    """Hyperbolic ReLU activation function for the Poincaré ball model.

    This activation function operates on hyperbolic space by:
    1. Mapping from the manifold to the tangent space at the origin (logmap0)
    2. Applying ReLU element-wise in the tangent space
    3. Mapping back to the manifold from the origin (expmap0)

    Args:
        x: Input ManifoldArray on the Poincaré ball manifold.
        c: Curvature of the Poincaré ball. If None, extracted from x.manifold.
        axis: The axis along which to perform the operation (default: -1).

    Returns:
        ManifoldArray with ReLU applied in tangent space and mapped back to manifold.

    References:
        Based on the approach used in hyperbolic neural networks where activation
        functions are applied in tangent space to maintain differentiability while
        preserving the hyperbolic geometry.
    """
    # Extract the underlying JAX array and manifold
    data = x.data
    manifold = x.manifold

    # Get curvature from manifold if not provided
    if c is None:
        if not hasattr(manifold, 'curvature'):
            raise ValueError(
                "Curvature not provided and manifold does not have curvature attribute"
            )
        c = manifold.curvature()

    # Step 1: Map from manifold to tangent space at origin
    tangent = logmap0(data, c, axis=axis)

    # Step 2: Apply ReLU in tangent space
    tangent_relu = jax.nn.relu(tangent)

    # Step 3: Map back to manifold from origin
    result = expmap0(tangent_relu, c, axis=axis)

    # Return as ManifoldArray
    return ManifoldArray(data=result, manifold=manifold)


def helu(
    x: ManifoldArray,
    alpha: float = 1.0,
    c: Optional[jax.Array] = None,
    axis: int = -1,
) -> ManifoldArray:
    """Hyperbolic ELU activation function for the Poincaré ball model.

    This activation function operates on hyperbolic space by:
    1. Mapping from the manifold to the tangent space at the origin (logmap0)
    2. Applying ELU element-wise in the tangent space
    3. Mapping back to the manifold from the origin (expmap0)

    Args:
        x: Input ManifoldArray on the Poincaré ball manifold.
        alpha: Coefficient for negative values (default: 1.0).
        c: Curvature of the Poincaré ball. If None, extracted from x.manifold.
        axis: The axis along which to perform the operation (default: -1).

    Returns:
        ManifoldArray with ELU applied in tangent space and mapped back to manifold.

    References:
        Based on the approach used in hyperbolic neural networks where activation
        functions are applied in tangent space to maintain differentiability while
        preserving the hyperbolic geometry.
    """
    # Extract the underlying JAX array and manifold
    data = x.data
    manifold = x.manifold

    # Get curvature from manifold if not provided
    if c is None:
        if not hasattr(manifold, 'curvature'):
            raise ValueError(
                "Curvature not provided and manifold does not have curvature attribute"
            )
        c = manifold.curvature()

    # Step 1: Map from manifold to tangent space at origin
    tangent = logmap0(data, c, axis=axis)

    # Step 2: Apply ELU in tangent space
    tangent_elu = jax.nn.elu(tangent, alpha=alpha)

    # Step 3: Map back to manifold from origin
    result = expmap0(tangent_elu, c, axis=axis)

    # Return as ManifoldArray
    return ManifoldArray(data=result, manifold=manifold)


class HReLU(nnx.Module):
    """Hyperbolic ReLU activation module for the Poincaré ball model.

    This is a stateless module that wraps the hrelu function for convenient use
    in neural network architectures.

    Example:
        >>> manifold = PoincareBall(c=1.0)
        >>> activation = HReLU()
        >>> x = ManifoldArray(data=jnp.array([[0.1, 0.2], [-0.1, 0.3]]), manifold=manifold)
        >>> y = activation(x)
    """

    def __init__(self, axis: int = -1):
        """Initialize the HReLU module.

        Args:
            axis: The axis along which to perform the operation (default: -1).
        """
        super().__init__()
        self.axis = axis

    def __call__(
        self, x: ManifoldArray, c: Optional[jax.Array] = None
    ) -> ManifoldArray:
        """Apply hyperbolic ReLU activation.

        Args:
            x: Input ManifoldArray on the Poincaré ball manifold.
            c: Optional curvature override. If None, extracted from x.manifold.

        Returns:
            ManifoldArray with HReLU applied.
        """
        return hrelu(x, c=c, axis=self.axis)


class HElu(nnx.Module):
    """Hyperbolic ELU activation module for the Poincaré ball model.

    This is a stateless module that wraps the helu function for convenient use
    in neural network architectures.

    Example:
        >>> manifold = PoincareBall(c=1.0)
        >>> activation = HElu(alpha=1.0)
        >>> x = ManifoldArray(data=jnp.array([[0.1, 0.2], [-0.1, 0.3]]), manifold=manifold)
        >>> y = activation(x)
    """

    def __init__(self, alpha: float = 1.0, axis: int = -1):
        """Initialize the HElu module.

        Args:
            alpha: Coefficient for negative values (default: 1.0).
            axis: The axis along which to perform the operation (default: -1).
        """
        super().__init__()
        self.alpha = alpha
        self.axis = axis

    def __call__(
        self, x: ManifoldArray, c: Optional[jax.Array] = None
    ) -> ManifoldArray:
        """Apply hyperbolic ELU activation.

        Args:
            x: Input ManifoldArray on the Poincaré ball manifold.
            c: Optional curvature override. If None, extracted from x.manifold.

        Returns:
            ManifoldArray with HElu applied.
        """
        return helu(x, alpha=self.alpha, c=c, axis=self.axis)
