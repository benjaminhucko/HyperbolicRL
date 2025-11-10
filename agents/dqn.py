import time

import optax
from flax import nnx
from flax.training import train_state
from flax.core import FrozenDict
import jax.numpy as jnp
import jax

from agents.agent import Agent
from neural_networks import ActorCritic, Critic




def direct_q_values(model, observations):
    q_values = model(observations)
    return q_values


class DQNPolicy(nnx.Module):
    def __init__(self, obs_shape, n_actions, rng, config):
        self.model = ActorCritic(obs_shape[-1], n_actions, rng, config)

    @staticmethod
    @nnx.jit
    def q_values(model, observations):
        advantages, values = model(observations)
        q_values = values + (advantages - advantages.mean())
        return q_values

    def __call__(self, observations, *args):
        advantages, values = self.model(observations)
        q_values = values + (advantages - advantages.mean())
        action = jnp.argmax(q_values, axis=-1)
        return action, {}

class DQNAgent(Agent):
    def __init__(self, obs_shape, n_actions, rngs, config):
        super().__init__(obs_shape, n_actions, rngs, config)

        self.target_model = ActorCritic(obs_shape[-1], n_actions, rngs, config)
        self.policy = DQNPolicy(obs_shape, n_actions, rngs, config)
        nnx.update(self.target_model, nnx.state(self.policy.model))
        self.optimizer = nnx.Optimizer(self.policy.model, optax.adam(learning_rate=self.config.learning_rate), wrt=nnx.Param)

    @staticmethod
    @nnx.jit
    def get_q_values(model, observations):
        advantages, values = model(observations)
        q_values = values + (advantages - advantages.mean())
        return q_values

    @staticmethod
    @nnx.jit
    def train_step(model, optimizer, targets, states, actions):
        def loss_fn(model):
            q_values = DQNAgent.get_q_values(model, states)
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

            q_values_behaviour = self.get_q_values(self.policy.model, next_states)
            target_actions = jnp.argmax(q_values_behaviour, axis=-1)

            q_values_target = self.get_q_values(self.target_model, next_states)
            q_values_next = jnp.take_along_axis(q_values_target, target_actions[:, None], axis=-1).squeeze(-1)

            targets = rewards + discounts * q_values_next

            td_errors = self.train_step(self.policy.model, self.optimizer, targets, states, actions)

            # Priority replay
            buffer.update_priorities(idxs, jnp.abs(td_errors))

            # DDQN
            new_target_params = optax.incremental_update(nnx.state(self.policy.model), nnx.state(self.target_model),
                                                         self.config.polyak_tau)
            nnx.update(self.target_model, new_target_params)