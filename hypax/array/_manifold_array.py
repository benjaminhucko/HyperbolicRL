import jax
import jax.numpy as jnp
from flax import struct

from hypax.manifolds import Manifold

@struct.dataclass
class ManifoldArray:
    data: jax.Array
    manifold: Manifold

    @property
    def ndim(self):
        return self.data.ndim

    def dim(self) -> int:
        """PyTorch-style alias used by some shared helpers."""
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    def flatten(self, manifold_axis, start_axis: int = 1, end_axis: int = -1):
        """Flattens tensor by reshaping it. If start_dim or end_dim are passed,
        only dimensions starting with start_dim and ending with end_dim are flattend.

        """
        new_data = self.manifold.flatten(self.data, manifold_axis, start_axis=start_axis, end_axis=end_axis)
        return self.replace(data=new_data)