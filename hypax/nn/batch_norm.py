from flax import nnx
import jax.numpy as jnp
from hypll.tensors import TangentTensor

from hypax.array import ManifoldArray
from hypax.manifolds import Manifold


class HBatchNorm(nnx.Module):
    def __init__(
        self,
        features: int,
        manifold: Manifold,
        use_midpoint: bool = False,
    ) -> None:
        super(HBatchNorm, self).__init__()
        self.features = features
        self.manifold = manifold
        self.use_midpoint = use_midpoint

        self.bias = nnx.Param(jnp.zeros(features))
        self.weight = nnx.Param(1.0)

    def forward(self, x: ManifoldArray) -> ManifoldArray:
        if self.use_midpoint:
            input_mean = self.manifold.midpoint(x=x, axis=1)
        else:
            input_mean = self.manifold.frechet_mean(x=x, axis=1)

        input_var = self.manifold.frechet_variance(x=x, mu=input_mean, axis=1)

        input_logm = self.manifold.transp(
            v=self.manifold.logmap(input_mean, x),
            y=bias_on_manifold,
        )

        input_logm.data = jnp.sqrt((self.weight / (input_var + 1e-6))) * input_logm.data

        output = self.manifold.expmap(input_logm)

        return output


class HBatchNorm2d(nnx.Module):
    """
    2D implementation of hyperbolic batch normalization.

    Based on:
        https://arxiv.org/abs/2003.00335
    """

    def __init__(
        self,
        features: int,
        manifold: Manifold,
        use_midpoint: bool = False,
    ) -> None:
        super(HBatchNorm2d, self).__init__()
        self.features = features
        self.manifold = manifold
        self.use_midpoint = use_midpoint

        self.norm = HBatchNorm(
            features=features,
            manifold=manifold,
            use_midpoint=use_midpoint,
        )

    def forward(self, x: ManifoldArray) -> ManifoldArray:
        flat_x = ManifoldArray(
            data=x.tensor.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2),
            manifold=x.manifold,
            man_dim=-1,
        )
        flat_x = self.norm(flat_x)
        new_tensor = flat_x.tensor.reshape(batch_size, height, width, self.features).permute(
            0, 3, 1, 2
        )
        return ManifoldArray(data=new_tensor, manifold=x.manifold)