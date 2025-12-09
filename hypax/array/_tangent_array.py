import jax
from flax import struct

@struct.dataclass
class TangentArray:
    data: jax.Array