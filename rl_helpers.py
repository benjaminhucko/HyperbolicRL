import jax.numpy as jnp
import jax
import optax
import rlax
import distrax


# all_values = jnp.concatenate((values.squeeze(), final_value[None, :]), axis=0)
# discounts = (1 - dones.squeeze()) * self.config.gamma
# batch_advantage_estimation = vmap(rlax.truncated_generalized_advantage_estimation,
#                                   in_axes=(1, 1, None, 1))
#
# advantages = batch_advantage_estimation(rewards, discounts, self.config.gae_lambda, all_values)

def _q_loss_fn(params, apply_fn, targets, obs, actions):
    q_values = apply_fn(params, obs)
    q_selected = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)
    loss = jnp.mean(optax.squared_error(q_selected, targets))
    return loss

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

    policy_loss = rlax.clipped_surrogate_pg_loss(ratio, advantages, clip_threshold)
    entropy_loss = jnp.mean(policy.entropy())
    value_loss = jnp.mean(optax.squared_error(new_values, returns))
    loss = policy_loss - regularization * entropy_loss + value_weight * value_loss

    return loss

def generalized_advantage_estimation_(values, rewards, dones, term_value, discount_factor, lambda_):
    def fold_left(last_gae, rest):
        td_error, discount = rest
        last_gae = td_error + discount * lambda_ * last_gae
        return last_gae, last_gae

    discounts = jnp.where(dones, 0, discount_factor)
    td_errors = rewards + discounts * jnp.append(values[1:], jnp.expand_dims(term_value, 0), axis=0) - values

    _, advantages = jax.lax.scan(fold_left, jnp.zeros(td_errors.shape[1]), (td_errors, discounts), reverse=True)
    return advantages

q_loss_fn = jax.jit(_q_loss_fn, static_argnames=['apply_fn'])
ppo_loss_fn = jax.jit(_ppo_loss_fn, static_argnames=['apply_fn', 'clip_threshold',
                                                     'regularization', 'value_weight'])
generalized_advantage_estimation = jax.jit(generalized_advantage_estimation_, static_argnames=['discount_factor',
                                                                                               'lambda_'])

