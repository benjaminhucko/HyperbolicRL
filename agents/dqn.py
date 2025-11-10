import time

import optax
from flax import nnx
from flax.training import train_state
from flax.core import FrozenDict
import jax.numpy as jnp
import jax

from agents.agent import Agent
from neural_networks import get_network_class

def direct_q_values(model, observation):
    q_values = model(observation)
    return q_values

def duelling_q_values(model, observation):
    advantages, values = model(observation)
    q_values = values + (advantages - advantages.mean())
    return q_values


class DQNPolicy(nnx.Module):
    def __init__(self, obs_shape, n_actions, rng, config):
        self.model = get_network_class(config)(obs_shape[-1], n_actions, rng, config)
        self.q_val_fn = nnx.static(duelling_q_values if config.duelling else direct_q_values)

    def __call__(self, observations, *args):
        q_values = self.q_val_fn(self.model, observations)
        action = jnp.argmax(q_values, axis=-1)
        return action, {}

class DQNAgent(Agent):
    def __init__(self, obs_shape, n_actions, rngs, config):
        super().__init__(obs_shape, n_actions, rngs, config)
        self.policy = DQNPolicy(obs_shape, n_actions, rngs, config)

        if config.ddqn:
            self.target_model = get_network_class(config)(obs_shape[-1], n_actions, rngs, config)
            nnx.update(self.target_model, nnx.state(self.policy.model))
        else:
            self.target_model = self.policy.model

        self.optimizer = nnx.Optimizer(self.policy.model, optax.adam(learning_rate=self.config.learning_rate), wrt=nnx.Param)

    @staticmethod
    @nnx.jit(static_argnames=['q_val_fn'])
    def train_step(model, optimizer, targets, states, actions, q_val_fn):
        def loss_fn(model):
            q_values = q_val_fn(model, states)
            q_selected = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)
            td_errors = q_selected - targets
            loss = jnp.mean(optax.squared_error(td_errors))
            return loss, {'td_errors': td_errors}

        grads, aux = nnx.grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)  # in-place updates
        return aux['td_errors']

    def update(self, buffer, rngs):
        # batches_per_epoch = buffer.size // self.config.batch_size
        for _ in range(self.config.n_epochs):
            states, actions, rewards, next_states, discounts, idxs = buffer.sample_batch(self.config.batch_size, rngs())

            next_actions, _ = self.policy(next_states)

            q_values_target = self.policy.q_val_fn(self.target_model, next_states)
            q_values_next = jnp.take_along_axis(q_values_target, next_actions[:, None], axis=-1).squeeze(-1)

            targets = rewards + discounts * q_values_next

            td_errors = self.train_step(self.policy.model, self.optimizer, targets, states, actions,
                                        self.policy.q_val_fn)

            # Priority replay
            buffer.update_priorities(idxs, jnp.abs(td_errors))

            # DDQN
            new_target_params = optax.incremental_update(nnx.state(self.policy.model), nnx.state(self.target_model),
                                                         self.config.polyak_tau)
            nnx.update(self.target_model, new_target_params)