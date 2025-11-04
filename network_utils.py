import distrax
import jax
import jax.numpy as jnp
import rlax
from flax.training import train_state
import optax

def create_train_state(rng_key, model, input_shape, config):
    params = model.init(rng_key, jnp.ones((1,) + input_shape))
    tx = optax.adam(config.lr)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def _q_loss_fn(params, apply_fn, targets, obs, actions):
    q_values = apply_fn(params, obs)
    q_selected = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)
    loss = jnp.mean(optax.squared_error(q_selected, targets))
    return loss

q_loss_fn = jax.jit(_q_loss_fn, static_argnames=['apply_fn'])

def _ppo_loss_fn(params, apply_fn, targets,
                 obs: jax.Array, actions, old_log_probs: jax.Array,
                 clip_threshold, regularization, value_weight):
    returns, advantages = targets
    result = apply_fn(params, obs)
    action_logits = result[:, :-1]
    new_values = result[:, -1]

    policy = distrax.Categorical(action_logits)
    log_probs = policy.log_prob(actions)
    log_ratio = log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)

    approx_kl = (-log_ratio).mean()
    jax.debug.print("ratio: {ratio} kl: {kl}", ratio=ratio.shape, kl=approx_kl)
    policy_loss = rlax.clipped_surrogate_pg_loss(ratio, advantages, clip_threshold)
    entropy_loss = jnp.mean(policy.entropy())
    value_loss = jnp.mean(optax.squared_error(new_values, returns))
    loss = policy_loss - regularization * entropy_loss + value_weight * value_loss
    jax.debug.print("policy_loss: {pl}, entropy_loss: {el}, value_loss: {vl}",
                    pl=policy_loss, el=entropy_loss, vl=value_loss)

    return loss

ppo_loss_fn = jax.jit(_ppo_loss_fn, static_argnames=['apply_fn', 'clip_threshold',
                                                     'regularization', 'value_weight'])

def train_step(state, loss_fn, y, *x):
    grads = jax.grad(loss_fn)(state.params, state.apply_fn, y, *x)
    return state.apply_gradients(grads=grads)

def network_forward(state, x):
    return state.apply_fn(state.params, x)
