from typing import final

import distrax
import jax
import optax
import jax.numpy as jnp
import rlax
from flax import nnx

from agents.agent import Agent
from neural_networks import ActorCritic

class PPOPolicy(nnx.Module):
    def __init__(self, obs_shape, n_actions, rngs, config):
        self.model = ActorCritic(obs_shape[-1], n_actions, rngs, config)

    def __call__(self, obs, key):
        action_logits, values = self.model(obs)
        policy = distrax.Categorical(logits=action_logits)
        action = policy.sample(seed=key)
        log_probs = policy.log_prob(action).squeeze()

        return action, {'log_probs': log_probs, 'values': values.squeeze()}


class PPOAgent(Agent):
    def __init__(self, obs_shape, n_actions, rngs, config):
        super().__init__(obs_shape, n_actions, rngs, config)
        self.policy = PPOPolicy(obs_shape, n_actions, rngs, config)
        self.optimizer = nnx.Optimizer(self.policy.model, optax.adam(learning_rate=self.config.learning_rate), wrt=nnx.Param)

    @staticmethod
    @nnx.jit
    def generalized_advantage_estimation(values, rewards, dones, term_value, discount_factor, lambda_):
        def fold_left(last_gae, rest):
            td_error, discount = rest
            last_gae = td_error + discount * lambda_ * last_gae
            return last_gae, last_gae

        discounts = jnp.where(dones, 0, discount_factor)

        td_errors = rewards + discounts * jnp.append(values[1:], jnp.expand_dims(term_value, 0), axis=0) - values

        _, advantages = jax.lax.scan(fold_left, jnp.zeros(td_errors.shape[1]), (td_errors, discounts), reverse=True)
        return advantages

    @staticmethod
    @nnx.jit
    def train_step(model, optimizer, returns, advantages, observations, actions, old_log_probs,
                   clip_threshold, regularization, value_weight):
        def ppo_loss(model):
            action_logits, values = model(observations)

            policy = distrax.Categorical(action_logits)
            log_probs = policy.log_prob(actions)
            log_ratio = log_probs - old_log_probs
            ratio = jnp.exp(log_ratio)

            policy_loss = rlax.clipped_surrogate_pg_loss(ratio, advantages, clip_threshold)
            entropy_loss = jnp.mean(policy.entropy())
            value_loss = jnp.mean(optax.squared_error(values.squeeze(), returns))

            loss = policy_loss - regularization * entropy_loss + value_weight * value_loss
            return loss

        grads = nnx.grad(ppo_loss)(model)
        optimizer.update(model, grads)  # in-place updates
        return {}

    def update(self, buffer, rng):
        obs, actions, rewards, dones, log_probs, values, final_obs = buffer.get()
        print(final_obs.shape)
        _, final_value = self.policy.model(final_obs)
        final_value = final_value.squeeze()

        # discounts = jnp.where(dones, 0, self.config.gamma)
        # adv_fn_rlax = jax.vmap(rlax.truncated_generalized_advantage_estimation, in_axes=(1, 1, None, 1), out_axes=1)
        # advantages = adv_fn_rlax(rewards, discounts, self.config.gae_lambda,
        #                          jnp.concatenate((values, jnp.expand_dims(final_value, 0)), 0))
        advantages = self.generalized_advantage_estimation(values, rewards, dones, final_value,
                                                           self.config.gamma, self.config.gae_lambda)
        advantages = advantages.reshape(-1)
        returns = advantages + values.reshape(-1)
        obs = jnp.reshape(obs, (-1, *obs.shape[2:]))
        actions = jnp.reshape(actions, -1)
        log_probs = jnp.reshape(log_probs, -1)
        epoch_size = obs.shape[0]
        for _ in range(self.config.n_epochs):
            epoch_indices = jax.random.permutation(rng(), epoch_size, independent=True)
            for start_idx in range(0, epoch_size, self.config.batch_size):
                end_idx = start_idx + self.config.batch_size
                idxs = epoch_indices[start_idx:end_idx]
                self.train_step(self.policy.model, self.optimizer, returns[idxs], advantages[idxs],
                                obs[idxs], actions[idxs], log_probs[idxs],
                                self.config.clip_threshold, self.config.entorpy_weight,
                                self.config.value_weight)

