from functools import partial

import jax
import jax.numpy as jnp
import optax
import rlax
from babel import support
from flax import nnx
from jax import vmap

from optimization.loss import c51_targets


# THIS OKAY
def select_actions(values, actions):
    greedy_values = jnp.take_along_axis(values,
                                        jnp.expand_dims(actions, axis=tuple(range(1, values.ndim))),
                                        axis=1).squeeze()
    return greedy_values

# loss functions

def q_loss_fn(actual, targets):
    td_errors = actual - targets
    loss = jnp.mean(optax.squared_error(td_errors))
    return loss, jnp.abs(td_errors)

def categorical_loss_fn(actual, targets):
    errors = optax.softmax_cross_entropy(actual, targets)
    loss = jnp.mean(errors)
    return loss, errors

# POST PROCESSING ON OUTPUT

def duelling_model_post(out):
    advantages, values = out
    post_out = values + (advantages - advantages.mean())
    return post_out

def duelling_dist_model_post(out, atoms, logits):
    advantages, values = out
    advantages = advantages.reshape((advantages.shape[0], -1, atoms))
    post_out = values[:, None] + (advantages - advantages.mean(axis=-1, keepdims=True))
    if not logits:
        post_out = nnx.softmax(post_out, axis=-1)
    return post_out

def direct_dist_model_post(out, atoms, logits):
    q_values = out
    post_out = q_values.reshape((q_values.shape[0], -1, atoms))

    if not logits:
        post_out = nnx.softmax(post_out, axis=-1)
    return post_out

# ------------------------
def get_loss(model, observations, actions, targets, key, model_output_fn, loss_fn):
    actual = model_output_fn(model, observations, key)
    selected_actual = select_actions(actual, actions)
    loss, errors = loss_fn(selected_actual, targets)
    return loss, {'errors': errors}

def get_q_values(model, observations, key, model_out_fn, q_fn):
    post_out = model_out_fn(model, observations, key)
    q_values = q_fn(post_out)
    return q_values

def get_model_outputs(model, observations, key, post_fn):
    out = model(observations, key)
    targets = post_fn(out)
    return targets

def make_output_fn(config, logits=False):
    if config.distributional and config.duelling:
        post_fn = partial(duelling_dist_model_post, atoms=config.atoms, logits=logits)
    elif config.distributional:
        post_fn = partial(direct_dist_model_post, atoms=config.atoms, logits=logits)
    elif config.duelling:
        post_fn = partial(duelling_model_post)
    else:
        post_fn = nnx.identity
    return partial(get_model_outputs, post_fn=post_fn)

def make_q_value_fn(config, support):
    model_post_out = make_output_fn(config)
    if config.distributional:
        q_fn = partial(jnp.dot, b=support)
    else:
        q_fn = nnx.identity
    return partial(get_q_values, model_out_fn=model_post_out, q_fn=q_fn)

def make_loss_fn(config):
    model_output_fn = make_output_fn(config, logits=True)
    if config.distributional:
        loss_fn = categorical_loss_fn
    else:
        loss_fn = q_loss_fn
    return partial(get_loss, model_output_fn=model_output_fn, loss_fn=loss_fn)


def classic_targets_fn(rewards, discounts, greedy_targets):
    targets = rewards + discounts * greedy_targets
    return targets

def make_targets_fn(config, support):
    if config.distributional:
        # targets = jnp.interp(self.support, target_support, target_prob_distribution)
        targets_fn = partial(c51_targets, support=support)
    else:
        targets_fn = classic_targets_fn
    return targets_fn
