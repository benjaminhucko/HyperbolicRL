from functools import partial

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import vmap


def select_actions(values, actions):
    greedy_values = jnp.take_along_axis(values,
                                        jnp.expand_dims(actions, axis=tuple(range(1, values.ndim))),
                                        axis=1).squeeze()
    return greedy_values

def project_distribution(supports, weights, target_support):
    """Projects a batch of (support, weights) onto target_support.

    Based on equation (7) in (Bellemare et al., 2017):
    https://arxiv.org/abs/1707.06887
    In the rest of the comments we will refer to this equation simply as Eq7.

    Args:
    supports: Jax array of shape (num_dims) defining supports for the
      distribution.
    weights: Jax array of shape (num_dims) defining weights on the original
      support points. Although for the CategoricalDQN agent these weights are
      probabilities, it is not required that they are.
    target_support: Jax array of shape (num_dims) defining support of the
      projected distribution. The values must be monotonically increasing. Vmin
      and Vmax will be inferred from the first and last elements of this Jax
      array, respectively. The values in this Jax array must be equally spaced.

    Returns:
    A Jax array of shape (num_dims) with the projection of a batch
    of (support, weights) onto target_support.

    Raises:
    ValueError: If target_support has no dimensions, or if shapes of supports,
      weights, and target_support are incompatible.
    """
    v_min, v_max = target_support[0], target_support[-1]
    # `N` in Eq7.
    num_dims = target_support.shape[0]
    # delta_z = `\Delta z` in Eq7.
    delta_z = (v_max - v_min) / (num_dims - 1)
    # clipped_support = `[\hat{T}_{z_j}]^{V_max}_{V_min}` in Eq7.
    clipped_support = jnp.clip(supports, v_min, v_max)
    # numerator = `|clipped_support - z_i|` in Eq7.
    numerator = jnp.abs(clipped_support - target_support[:, None])
    quotient = 1 - (numerator / delta_z)
    # clipped_quotient = `[1 - numerator / (\Delta z)]_0^1` in Eq7.
    clipped_quotient = jnp.clip(quotient, 0, 1)
    # inner_prod = `\sum_{j=0}^{N-1} clipped_quotient * p_j(x', \pi(x'))` in Eq7.
    inner_prod = clipped_quotient * weights
    return jnp.squeeze(jnp.sum(inner_prod, -1))

def q_loss_fn(actual, targets):
    td_errors = actual - targets
    loss = jnp.mean(optax.squared_error(td_errors))
    return loss, jnp.abs(td_errors)

def categorical_loss_fn(actual, targets):
    actual_distribution = distrax.Categorical(probs=actual)
    target_distribution = distrax.Categorical(probs=targets)
    errors = actual_distribution.kl_divergence(target_distribution)
    loss = jnp.mean(errors)
    return loss, errors

def direct_model_post(out):
    return out

def duelling_model_post(out):
    values, advantages = out
    post_out = values + (advantages - advantages.mean())
    return post_out

def duelling_dist_model_post(out, atoms):
    values, advantages = out
    advantages = advantages.reshape((advantages.shape[0], atoms, -1))
    q_values = values + (advantages - advantages.mean(axis=-1, keepdims=True))
    post_out = nnx.softmax(q_values, axis=-1)
    return post_out

def direct_dist_model_post(out, atoms):
    q_values = out
    q_values = q_values.reshape((q_values.shape[0], -1, atoms))
    post_out = nnx.softmax(q_values, axis=1)
    return post_out

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

def make_output_fn(config):
    if config.distributional and config.duelling:
        post_fn = partial(duelling_model_post, atoms=config.atoms)
    elif config.distributional:
        post_fn = partial(direct_dist_model_post, atoms=config.atoms)
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
    model_output_fn = make_output_fn(config)
    if config.distributional:
        loss_fn = categorical_loss_fn
    else:
        loss_fn = q_loss_fn
    return partial(get_loss, model_output_fn=model_output_fn, loss_fn=loss_fn)

def distributional_targets_fn(rewards, discounts, target_prob_distribution, support):
    target_support = rewards[:, None] + jnp.outer(discounts, support)
    target_support = target_support.clip(min=support[0], max=support[-1])
    batched_project_fn = vmap(project_distribution, in_axes=(0, 0, None))
    targets = batched_project_fn(target_support, target_prob_distribution, support)
    return targets

def classic_targets_fn(rewards, discounts, greedy_targets):
    targets = rewards + discounts * greedy_targets
    return targets

def make_targets_fn(config, support):
    if config.distributional:
        # targets = jnp.interp(self.support, target_support, target_prob_distribution)
        targets_fn = partial(distributional_targets_fn, support=support)
    else:
        targets_fn = classic_targets_fn
    return targets_fn
