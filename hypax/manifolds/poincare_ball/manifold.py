# manifold.py
#
# JAX implementation of the Poincaré ball model of hyperbolic space

from __future__ import annotations

import functools
import typing as tp

import jax
import jax.numpy as jnp
from hypax.manifolds._base import Manifold
from hypax.manifolds.curvature import Curvature
from hypax.manifolds.poincare_ball._diffgeom import (
    expmap0,
    expmap,
    logmap0,
    logmap,
    project,
    mobius_add,
    dist,
    inner as poincare_inner,
    euc_to_tangent as poincare_euc_to_tangent,
    transp as poincare_transp,
    cdist as poincare_cdist,
)
from hypax.manifolds.poincare_ball._linalg import poincare_fully_connected
from hypax.manifolds.poincare_ball._stats import (
    frechet_mean as poincare_frechet_mean,
    midpoint as poincare_midpoint,
)
from hypax.utils.math import beta_func


class PoincareBall(Manifold):
    """Poincaré ball model of hyperbolic space.

    The Poincaré ball is a conformal model of hyperbolic geometry. Points on
    the manifold are represented as vectors x with ||x|| < 1/√c, where c is
    the curvature.

    Attributes:
        curvature: Curvature of the manifold (scalar > 0). Higher curvature means
                   more "curved" space with a smaller radius.

    Example:
        >>> manifold = PoincareBall(c=1.0)
        >>> # Points must satisfy ||x|| < 1/√c
        >>> x = jnp.array([[0.1, 0.2], [0.3, -0.1]])
        >>> # Use manifold operations like expmap, logmap, etc.
    """

    def __init__(self, curvature: Curvature):
        """Initialize PoincareBall manifold.

        Args:
            c: Curvature parameter (default: 1.0). Must be positive.
               Can be a scalar float or JAX array.
        """
        super().__init__()
        self.curvature = curvature

    def expmap(
        self, v: jax.Array, x: jax.Array | None = None, axis: int = -1
    ) -> jax.Array:
        """Exponential map: map from tangent space to manifold.

        Args:
            v: Tangent vector
            x: Base point on manifold (if None, uses origin)
            axis: Manifold dimension axis

        Returns:
            Point on manifold
        """
        if x is None:
            return expmap0(v, self.curvature(), axis=axis)
        else:
            return expmap(x, v, self.curvature(), axis=axis)

    def logmap(
        self, y: jax.Array, x: jax.Array | None = None, axis: int = -1
    ) -> jax.Array:
        """Logarithmic map: map from manifold to tangent space.

        Args:
            y: Point on manifold
            x: Base point on manifold (if None, uses origin)
            axis: Manifold dimension axis

        Returns:
            Tangent vector at base point
        """
        if x is None:
            return logmap0(y, self.curvature(), axis=axis)
        else:
            return logmap(x, y, self.curvature(), axis=axis)

    def project(self, x: jax.Array, axis: int = -1, eps: float = -1.0) -> jax.Array:
        """Project point to be within the Poincaré ball.

        Args:
            x: Point (possibly outside ball)
            axis: Manifold dimension axis
            eps: Epsilon for numerical stability (if < 0, uses default)

        Returns:
            Point projected to be within ball
        """
        return project(x, self.curvature(), axis=axis, eps=eps)

    def mobius_add(self, x: jax.Array, y: jax.Array, axis: int = -1) -> jax.Array:
        """Möbius addition in the Poincaré ball.

        Args:
            x: First point on manifold
            y: Second point on manifold
            axis: Manifold dimension axis

        Returns:
            Result of Möbius addition
        """
        return mobius_add(x, y, self.curvature(), axis=axis)

    def dist(
        self, x: jax.Array, y: jax.Array, axis: int = -1, keepdims: bool = False
    ) -> jax.Array:
        """Hyperbolic distance between points.

        Args:
            x: First point on manifold
            y: Second point on manifold
            axis: Manifold dimension axis
            keepdims: Whether to keep dimensions

        Returns:
            Hyperbolic distance
        """
        return dist(x, y, self.curvature(), axis=axis, keepdims=keepdims)

    def cdist(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Pairwise hyperbolic distances between batches of points."""
        return poincare_cdist(x, y, self.curvature())

    def frechet_mean(
        self,
        x: jax.Array,
        *,
        reduce_axis: int,
        axis: int = -1,
        keepdims: bool = False,
        max_iter: int = 100,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> jax.Array:
        """Compute the Fréchet mean of points along ``reduce_axis``."""
        return poincare_frechet_mean(
            x=x,
            c=self.curvature(),
            manifold_axis=axis,
            reduce_axis=reduce_axis,
            keepdims=keepdims,
            max_iter=max_iter,
            rtol=rtol,
            atol=atol,
        )

    def midpoint(
        self,
        x: jax.Array,
        *,
        reduce_axis: int,
        axis: int = -1,
        keepdims: bool = False,
    ) -> jax.Array:
        """Compute the hyperbolic midpoint along ``reduce_axis``."""
        return poincare_midpoint(
            x=x,
            c=self.curvature(),
            manifold_axis=axis,
            reduce_axis=reduce_axis,
            keepdims=keepdims,
        )

    def inner(
        self,
        x: jax.Array,
        u: jax.Array,
        v: jax.Array,
        axis: int = -1,
        keepdims: bool = False,
    ) -> jax.Array:
        """Riemannian inner product at point ``x`` between tangent vectors ``u`` and ``v``."""

        return poincare_inner(
            x=x,
            u=u,
            v=v,
            c=self.curvature(),
            axis=axis,
            keepdims=keepdims,
        )

    def euc_to_tangent(
        self, x: jax.Array, u: jax.Array, axis: int = -1
    ) -> jax.Array:
        """Project Euclidean tensor ``u`` onto the tangent space at ``x``."""

        return poincare_euc_to_tangent(
            x=x,
            u=u,
            c=self.curvature(),
            axis=axis,
        )

    def transp(
        self, x: jax.Array, y: jax.Array, v: jax.Array, axis: int = -1
    ) -> jax.Array:
        """Parallel transport ``v`` from tangent space at ``x`` to tangent space at ``y``."""

        return poincare_transp(
            x=x,
            y=y,
            v=v,
            c=self.curvature(),
            axis=axis,
        )

    def construct_dl_parameters(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        key_z: jax.Array | None = None,
        key_bias: jax.Array | None = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> tp.Tuple[jax.Array, jax.Array | None]:
        """Construct and initialize parameters for deep learning layers.

        This method initializes weights and biases for hyperbolic neural network
        layers following the initialization strategy from HNN++:
        - For square/tall matrices (in_features <= out_features): identity init
        - For wide matrices (in_features > out_features): normal init

        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to create bias parameters
            key_z: JAX random key for weight initialization
            key_bias: JAX random key for bias initialization
            dtype: Data type for parameters

        Returns:
            Tuple of (weights, bias) where:
            - weights: Array of shape (in_features, out_features)
            - bias: Array of shape (out_features,) if bias=True, else None

        References:
            Chen et al. "Fully Hyperbolic Neural Networks" (HNN++), ACL 2022
        """
        if key_z is None:
            key_z = jax.random.PRNGKey(0)

        # Initialize weights using identity or normal initialization
        if in_features <= out_features:
            # Identity initialization for square or tall matrices
            weights = 0.5 * jnp.eye(in_features, out_features, dtype=dtype)
        else:
            # HNN++ initialization for wide matrices
            std = (2 * in_features * out_features) ** -0.5
            weights = (
                jax.random.normal(key_z, shape=(in_features, out_features), dtype=dtype)
                * std
            )

        # Initialize bias to zeros if needed
        if bias:
            bias_value = jnp.zeros(out_features, dtype=dtype)
        else:
            bias_value = None

        return weights, bias_value

    def fully_connected(
        self,
        x: jax.Array,
        z: jax.Array,
        bias: jax.Array | None,
        axis: int = -1,
    ) -> jax.Array:
        """Hyperbolic fully connected layer operation.

        Applies the fully connected transformation in hyperbolic space using
        hyperplane distances and projections.

        Args:
            x: Input array on the manifold
            z: Weight matrix (tangent space)
            bias: Bias vector (if None, no bias)
            axis: Manifold dimension axis

        Returns:
            Output array on the manifold

        References:
            Chen et al. "Fully Hyperbolic Neural Networks" (HNN++), ACL 2022
        """
        result = poincare_fully_connected(
            x=x, z=z, bias=bias, c=self.curvature(), axis=axis
        )
        # Project result to ensure it stays in the ball
        return self.project(result, axis=axis)

    def flatten(self, x, manifold_axis, start_axis=1, end_axis=-1):
        manifold_axis = x.ndim + manifold_axis if manifold_axis < 0 else manifold_axis
        start_axis = x.ndim + start_axis if start_axis < 0 else start_axis
        end_axis = x.ndim + end_axis if end_axis < 0 else end_axis

        flattened_shape = functools.reduce(lambda a, b: a * b, x.shape[start_axis: end_axis+1])
        new_shape = x.shape[:start_axis] + (flattened_shape, ) + x.shape[end_axis+1:]
        if start_axis <= manifold_axis <= end_axis:
            # Use beta concatenation to flatten the manifold dimension of the tensor.
            # Start by applying logmap at the origin and computing the betas.
            x = self.logmap(x, axis=manifold_axis)
            n_i = x.shape[manifold_axis]
            n = flattened_shape
            beta_n = beta_func(n / 2, 0.5)
            beta_n_i = beta_func(n_i / 2, 0.5)
            # Flatten the tensor and rescale.
            x = jnp.reshape(x, new_shape)
            x *= beta_n / beta_n_i
            new_manifold_axis = start_axis
            # Apply exponential map at the origin.
            x = self.expmap(x, axis=new_manifold_axis)
            return x
        else:
            flattened = jnp.reshape(x, new_shape)
            return flattened