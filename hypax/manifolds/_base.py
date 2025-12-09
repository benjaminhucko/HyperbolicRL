from abc import ABC, abstractmethod
from flax import nnx
import jax
import jax.numpy as jnp
import typing as tp

class Manifold(nnx.Module, ABC):
    """Abstract base class for Riemannian manifolds.

    This class defines the interface that all manifold implementations must follow.
    Manifolds provide the geometric operations needed for neural networks operating
    in non-Euclidean spaces.
    """

    @abstractmethod
    def expmap(self, v: jax.Array, x: jax.Array | None = None, axis: int = -1) -> jax.Array:
        """Exponential map: tangent space → manifold.

        Args:
            v: Tangent vector
            x: Base point (if None, uses origin)
            axis: Manifold axis

        Returns:
            Point on manifold
        """
        ...

    @abstractmethod
    def logmap(self, y: jax.Array, x: jax.Array | None = None, axis: int = -1) -> jax.Array:
        """Logarithmic map: manifold → tangent space.

        Args:
            y: Point on manifold
            x: Base point (if None, uses origin)
            axis: Manifold axis

        Returns:
            Tangent vector
        """
        ...

    @abstractmethod
    def project(self, x: jax.Array, axis: int = -1, eps: float = -1.0) -> jax.Array:
        """Project point onto the manifold.

        Args:
            x: Point (possibly off manifold)
            axis: Manifold axis
            eps: Numerical stability epsilon

        Returns:
            Point on manifold
        """
        ...

    @abstractmethod
    def dist(self, x: jax.Array, y: jax.Array, axis: int = -1, keepdims: bool = False) -> jax.Array:
        """Distance between points on the manifold.

        Args:
            x: First point
            y: Second point
            axis: Manifold axis
            keepdims: Keep dimensions

        Returns:
            Distance
        """
        ...

    @abstractmethod
    def construct_dl_parameters(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        key_z: jax.Array | None = None,
        key_bias: jax.Array | None = None,
        dtype: jnp.dtype = jnp.float32,
    ) -> tp.Tuple[jax.Array, jax.Array | None]:
        """Construct parameters for deep learning layers.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Include bias
            key_z: Random key for weights
            key_bias: Random key for bias
            dtype: Data type

        Returns:
            (weights, bias) tuple
        """
        ...

    @abstractmethod
    def fully_connected(
        self,
        x: jax.Array,
        z: jax.Array,
        bias: jax.Array | None,
        axis: int = -1,
    ) -> jax.Array:
        """Fully connected layer on the manifold.

        Args:
            x: Input
            z: Weights
            bias: Bias (optional)
            axis: Manifold axis

        Returns:
            Output on manifold
        """
        ...
