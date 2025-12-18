from flax import nnx
import jax
import typing as tp
import jax.numpy as jnp

class Curvature(nnx.Module):
    def __init__(self, value: float | jax.Array = 1.0,
                 constraining_strategy: tp.Callable[[jax.Array], jax.Array] = nnx.softplus,
                 learnable: bool = True,
                 param_dtype: jnp.dtype = jnp.float32):
        """Class representing curvature of a manifold.

            Attributes:
                value:
                    Learnable parameter indicating curvature of the manifold. The actual
                    curvature is calculated as constraining_strategy(value).
                constraining_strategy:
                    Function applied to the curvature value in order to constrain the
                    curvature of the manifold. By default uses softplus to guarantee
                    positive curvature.
            """
        if jnp.any(value <= 0):
            raise ValueError(f"Curvature must be positive, got {value}")
        value = jnp.asarray(value, dtype=param_dtype)
        self.learnable = nnx.static(learnable)
        if learnable:
            self.value = nnx.Param(value)
            self.constraining_strategy = constraining_strategy
            print('learnable parameter indicating curvature')
        else:
            self.value = nnx.static(value.item())
            self.constraining_strategy = nnx.identity

    def __call__(self):
        if self.learnable:
            curvature = self.value.value
        else:
            curvature = self.value
        return self.constraining_strategy(curvature)