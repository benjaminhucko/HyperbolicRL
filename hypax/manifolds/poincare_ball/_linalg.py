import jax
import jax.numpy as jnp
from typing import Tuple, Sequence

from hypax.manifolds.poincare_ball._diffgeom import logmap0, expmap0
from hypax.utils.math import beta_func

from hypax.manifolds.poincare_ball._stats import safe_norm


def _pair(value: int | Tuple[int, int] | Sequence[int]) -> Tuple[int, int]:
    """Normalize kernel/stride/padding arguments to 2-tuples."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected tuple of length 2, got {value}")
        return tuple(int(v) for v in value)
    if isinstance(value, Sequence):
        value = list(value)
        if len(value) != 2:
            raise ValueError(f"Expected sequence of length 2, got {value}")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def poincare_hyperplane_dists(
    x: jax.Array,
    z: jax.Array,
    r: jax.Array | None,
    c: jax.Array,
    axis: int = -1,
) -> jax.Array:
    """The Poincare signed distance to hyperplanes operation.

    Args:
        x (jax.Array): The input values.
        z (jax.Array): The hyperbolic vectors describing the hyperplane orientations
        r (jax.Array | None): The hyperplane offsets
        c (jax.Array): The curvature of the Poincare disk.
        dim (int, optional): The axis. Defaults to -1.

    Returns:
        jax.Array: signed distances of input w.r.t. the hyperplanes, denoted by v_k(x) in the HNN++ paper
    """

    axis_shifted_x = jnp.moveaxis(x, source=axis, destination=-1)

    c_sqrt = jnp.sqrt(c)
    lam = 2 / (1 - c * jnp.pow(axis_shifted_x, 2).sum(axis=-1, keepdims=True))
    z_norm = safe_norm(z, axis=0)

    # Computation can be simplified if there is no offset
    if r is None:
        dim_shifted_output = (
            2
            * z_norm
            / c_sqrt
            * jnp.asinh(c_sqrt * lam / z_norm * jnp.matmul(axis_shifted_x, z))
        )
    else:
        two_csqrt_r = 2.0 * c_sqrt * r
        dim_shifted_output = (
            2
            * z_norm
            / c_sqrt
            * jnp.asinh(
                c_sqrt
                * lam
                / z_norm
                * jnp.matmul(axis_shifted_x, z)
                * jnp.cosh(two_csqrt_r)
                - (lam - 1) * jnp.sinh(two_csqrt_r)
            )
        )

    return jnp.moveaxis(dim_shifted_output, source=-1, destination=axis)


def poincare_fully_connected(
    x: jax.Array,
    z: jax.Array,
    bias: jax.Array | None,
    c: jax.Array,
    axis: int = -1,
) -> jax.Array:
    """The Poincare fully connected layer operation.

    Args:
        x (jax.Array): The layer inputs
        z (jax.Array): The hyperbolic vectors describing the hyperplane orientations
        bias (jax.Array | None): The layer biases (hyperplane offsets)
        c (jax.Array): The curvature of the Poincare disk.
        axis (int, optional): The axis. Defaults to -1.

    Returns:
        jax.Array: Poincare FC transformed hyperbolic array, commonly denoted by y
    """
    c_sqrt = jnp.sqrt(c)
    x = poincare_hyperplane_dists(x=x, z=z, r=bias, c=c, axis=axis)
    x = jnp.sinh(c_sqrt * x) / c_sqrt
    return x / (1 + jnp.sqrt(1 + c * jnp.pow(x, 2).sum(axis=axis, keepdims=True)))


def unfold_2d(
    x: jax.Array,
    kernel_size: Tuple[int, int],
    stride: int | Tuple[int, int] = 1,
    padding: int | Tuple[int, int] = 0,
) -> jax.Array:
    """Extract sliding local blocks from a batched 2D input (im2col operation).

    This is the JAX equivalent of PyTorch's torch.nn.functional.unfold.

    Args:
        x: Input array with shape [batch, channels, height, width]
        kernel_size: Size of the sliding blocks as (kernel_h, kernel_w)
        stride: Stride of the sliding blocks (default: 1)
        padding: Implicit zero padding on both sides (default: 0)

    Returns:
        Array with shape [batch, channels * kernel_h * kernel_w, num_patches]
        where num_patches = out_h * out_w
    """
    batch_size, channels, height, width = x.shape
    kernel_h, kernel_w = kernel_size

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)

    # Apply padding if needed
    if pad_h > 0 or pad_w > 0:
        x = jnp.pad(
            x,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
            constant_values=0,
        )
        height += 2 * pad_h
        width += 2 * pad_w

    # Calculate output dimensions
    out_h = (height - kernel_h) // stride_h + 1
    out_w = (width - kernel_w) // stride_w + 1

    # Extract patches using strided slicing
    # Create indices for all patches
    patches = []
    # TODO reimplement with jax.lax.scan/batchify
    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride_h
            w_start = j * stride_w
            patch = x[:, :, h_start : h_start + kernel_h, w_start : w_start + kernel_w]
            # Reshape to [batch, channels * kernel_h * kernel_w]
            patch = patch.reshape(batch_size, -1)
            patches.append(patch)

    # Stack all patches: [batch, channels * kernel_h * kernel_w, num_patches]
    output = jnp.stack(patches, axis=-1)

    return output


def poincare_unfold(
    x: jax.Array,
    kernel_size: Tuple[int, int],
    in_channels: int,
    c: jax.Array,
    stride: int | Tuple[int, int] = 1,
    padding: int | Tuple[int, int] = 0,
    axis: int = 1,
) -> jax.Array:
    """Hyperbolic unfold operation for 2D convolution in the Poincare ball model.

    This operation extracts patches from the input and applies beta-concatenation
    to properly combine hyperbolic vectors when increasing dimensionality.

    The operation:
    1. Maps input from manifold to tangent space at origin (logmap0)
    2. Applies beta-concatenation rescaling to maintain hyperbolic geometry
    3. Extracts patches using standard unfold (im2col)
    4. Maps result back to manifold (expmap0)

    Args:
        x: Input array with shape [batch, channels, height, width]
        kernel_size: Size of the convolving kernel as (kernel_h, kernel_w)
        in_channels: Number of input channels
        c: Curvature of the Poincare ball
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides of the input (default: 0)
        axis: The manifold axis (default: 1 for channels)

    Returns:
        Unfolded array with shape [batch, kernel_vol * in_channels, num_patches]
        where kernel_vol = kernel_h * kernel_w and num_patches = out_h * out_w

    References:
        Chen et al. "Fully Hyperbolic Neural Networks" (HNN++), ACL 2022
        Beta-concatenation is used to properly rescale vectors when concatenating
        in hyperbolic space to preserve geometric properties.
    """
    kernel_h, kernel_w = kernel_size
    kernel_vol = kernel_h * kernel_w

    # Step 1: Map to tangent space at origin
    x_tangent = logmap0(x, c, axis=axis)

    # Step 2: Apply beta-concatenation rescaling
    # When concatenating vectors in hyperbolic space, we need to rescale
    # beta_ni corresponds to the original dimension (in_channels)
    # beta_n corresponds to the new dimension (in_channels * kernel_vol)
    beta_ni = beta_func(in_channels / 2, 1 / 2)
    beta_n = beta_func(in_channels * kernel_vol / 2, 1 / 2)
    rescale_factor = beta_n / beta_ni
    x_tangent = x_tangent * rescale_factor

    # Step 3: Apply Euclidean unfold in tangent space
    x_unfolded = unfold_2d(x_tangent, kernel_size, stride=stride, padding=padding)

    # Step 4: Map back to manifold
    x_manifold = expmap0(x_unfolded, c, axis=axis)

    return x_manifold
