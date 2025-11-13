from functools import partial

import distrax
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import vmap

def project_distribution2(supports, weights, target_support):
    v_min, v_max = target_support[0], target_support[-1]
    N = target_support.shape[0]
    delta_z = (v_max - v_min) / (N - 1)
    clipped_support = jnp.clip(supports, v_min, v_max)
    b = (clipped_support - v_min) / delta_z
    l = jnp.floor(b).astype(jnp.int32)
    u = jnp.ceil(b).astype(jnp.int32)

    l = jnp.clip(l, 0, N - 1)
    u = jnp.clip(u, 0, N - 1)

    m = jnp.zeros_like(target_support)
    m = m.at[l].add(weights * (u - b))
    m = m.at[u].add(weights * (b - l))
    m_sum = jnp.sum(m)
    m = jnp.where(m_sum > 0, m / m_sum, jnp.ones_like(m) / m.size)
    return m # normalize

def select_actions(values, actions):
    greedy_values = jnp.take_along_axis(values,
                                        jnp.expand_dims(actions, axis=tuple(range(1, values.ndim))),
                                        axis=1).squeeze()
    return greedy_values

def project_distribution(projected_from, weights, projected_to):
    v_min, v_max = projected_to[0], projected_to[-1]
    num_dims = projected_from.shape[0]
    delta_z = (v_max - v_min) / (num_dims - 1)
    distance = jnp.abs(projected_from[:, None] - projected_to[None, :])
    contribution = jnp.clip(1 - distance / delta_z, 0.0, 1.0)
    projected = jnp.sum(contribution * weights[:, None], axis=0)
    projected /= jnp.sum(projected)
    return projected

def q_loss_fn(actual, targets):
    td_errors = actual - targets
    loss = jnp.mean(optax.squared_error(td_errors))
    return loss, jnp.abs(td_errors)

def categorical_loss_fn(actual, targets):
    errors = optax.softmax_cross_entropy(actual, targets)
    loss = jnp.mean(errors)
    return loss, errors

def direct_model_post(out):
    return out

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

def distributional_targets_fn(rewards, discounts, target_prob_distribution, support):
    target_support = rewards[:, None] + jnp.outer(discounts, support)
    v_min, v_max = support[0], support[-1]
    target_support = jnp.clip(target_support, v_min, v_max)
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
