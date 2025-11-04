import jax
import jax.numpy as jnp
import optax

def network_init(rng_key, model, input_shape, config):
    params = model.init(rng_key, jnp.ones((1,) + input_shape))
    tx = optax.adam(config.lr)
    return params, tx

def train_step(state, loss_fn, y, *x):
    grads = jax.grad(loss_fn)(state.params, state.apply_fn, y, *x)
    return state.apply_gradients(grads=grads)

def network_forward(state, x):
    return state.apply_fn(state.params, x)
