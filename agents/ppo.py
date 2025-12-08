import time
from collections import defaultdict
from functools import partial

import distrax
import jax
import optax
import jax.numpy as jnp
import rlax
from flax import nnx
from flax.nnx import vmap

from agents.agent import Agent
from analysis.analyzer import Analyzer
from networks.euclidean import ActorCritic
from networks.factory import make_network, make_optim
from optimization.loss import hl_gauss_transform
from optimization.norm import normalize

def mse_value_loss(values, returns):
    value_loss = optax.squared_error(values.squeeze(), returns)
    return jnp.mean(value_loss)

def categorical_value_loss(values, returns, support):
    to_probs, fp = hl_gauss_transform(support)

    target_prob = to_probs(returns)
    value_loss = optax.softmax_cross_entropy(values, target_prob)
    return jnp.mean(value_loss)

def categorical_value_post(values, support):
    _, from_probs = hl_gauss_transform(support)
    values = nnx.softmax(values, axis=-1)
    return from_probs(values)


class PPOPolicy(nnx.Module):
    def __init__(self, n_channels, n_actions, post_fn, rngs, config):
        self.model = make_network(n_channels, n_actions, rngs, config)
        self.post_fn = post_fn

    def __call__(self, obs, key):
        network_key, sample_key = jax.random.split(key, 2)
        action_logits, values = self.model(obs, network_key)
        values = self.post_fn(values)
        policy = distrax.Categorical(logits=action_logits)
        action = policy.sample(seed=sample_key)
        log_probs = policy.log_prob(action).squeeze()

        return action, {'log_probs': log_probs, 'values': values.squeeze()}


class PPOAgent(Agent):
    def __init__(self, n_channels, n_actions, rngs, config):
        super().__init__(n_channels, n_actions, rngs, config)
        if config.distributional:
            self.support = jnp.linspace(config.v_min, config.v_max, config.atoms + 1)
            self.value_loss_fn = partial(categorical_value_loss, support=self.support)
            self.post_fn = partial(categorical_value_post, support=self.support)
        else:
            self.value_loss_fn = mse_value_loss
            self.post_fn = nnx.identity

        self.policy = PPOPolicy(n_channels, n_actions, self.post_fn, rngs, config)
        self.optimizer = make_optim(self.policy.model, config)

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
    @nnx.jit(static_argnames=['value_loss_fn', 'clip_threshold', 'value_weight', 'regularization',
                              'analyze', 'eval_grads'])
    def train_step(model, optimizer, returns, advantages, observations, actions, old_log_probs,
                   clip_threshold, regularization, value_weight, value_loss_fn,
                   analyze=False, eval_grads=False):
        def ppo_loss(model):
            out = model(observations, None, analyze=analyze)
            action_logits, values = out[0], out[1]

            policy = distrax.Categorical(action_logits)
            log_probs = policy.log_prob(actions)
            log_ratio = log_probs - old_log_probs
            ratio = jnp.exp(log_ratio)
            normalized_advantages = normalize(advantages)
            policy_loss = rlax.clipped_surrogate_pg_loss(ratio, normalized_advantages, clip_threshold)
            entropy_loss = jnp.mean(policy.entropy())
            value_loss = value_loss_fn(values, returns)
            loss = policy_loss - regularization * entropy_loss + value_weight * value_loss

            aux = {'policy_loss': policy_loss, 'value_loss': value_loss}
            if analyze:
                aux['embeddings'] = out[2]
            return loss, aux

        (loss, aux), grads = nnx.value_and_grad(ppo_loss, has_aux=True)(model)
        if eval_grads:
            return grads

        optimizer.update(model, grads)  # in-place updates

        if hasattr(model, 'manifold'):
            aux['curvature'] = model.manifold.curvature()
        aux['loss'] = loss
        return aux

    def update(self, buffer, rng, grad_analysis=False):
        obs, actions, rewards, dones, log_probs, values, final_obs = buffer.get()
        _, final_value = self.policy.model(final_obs, rng())
        final_value = self.post_fn(final_value).squeeze()
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
        stats = defaultdict(list)
        for _ in range(self.config.epochs):
            stats = defaultdict(list)

            epoch_indices = jax.random.permutation(rng(), epoch_size, independent=True)
            for start_idx in range(0, epoch_size, self.config.batch_size):
                end_idx = start_idx + self.config.batch_size
                idxs = epoch_indices[start_idx:end_idx]
                aux = self.train_step(self.policy.model, self.optimizer, returns[idxs], advantages[idxs],
                                      obs[idxs], actions[idxs], log_probs[idxs],
                                      self.config.clip_threshold, self.config.entorpy_weight,
                                      self.config.value_weight, self.value_loss_fn,
                                      analyze=self.config.analyze)

                stats = Analyzer.append(stats, aux)

        if grad_analysis:
            obs, actions, log_probs, advantages \
                = (jnp.expand_dims(val, axis=1) for val in [obs, actions, log_probs, advantages])
            for idx in range(epoch_size):
                grads = self.train_step(self.policy.model, self.optimizer, returns[idx], advantages[idx],
                                        obs[idx], actions[idx], log_probs[idx],
                                        self.config.clip_threshold, self.config.entorpy_weight,
                                        self.config.value_weight, self.value_loss_fn,
                                        analyze=False, eval_grads=True)

                batch_grads = Analyzer.process_raw_grads(grads)
                stats = Analyzer.append(stats, {'grads': batch_grads})
        return stats

    def behavioral_policy(self):
        return self.policy
