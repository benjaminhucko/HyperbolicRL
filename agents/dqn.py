import jax.numpy as jnp
import optax
from flax import nnx

from agents.agent import Agent
from agents.dqn_needs_refactor import make_q_value_fn, make_output_fn, make_loss_fn, make_targets_fn, select_actions
from agents.random_policy import EpsilonGreedyPolicy
from networks.factory import make_network, make_optim


class DQNPolicy(nnx.Module):
    def __init__(self, n_channels, n_actions, rng, config, support):
        self.model = make_network(n_channels, n_actions, rng, config)
        self.q_value_fn = nnx.static(make_q_value_fn(config, support))

    def __call__(self, observations, key):
        q_values = self.q_value_fn(self.model, observations, key)
        action = jnp.argmax(q_values, axis=-1)
        # jax.debug.print('q_values {q}', q=q_values)
        return action, {}

class DQNAgent(Agent):
    def __init__(self, n_channels, n_actions, rngs, config):
        super().__init__(n_channels, n_actions, rngs, config)
        self.n_actions = n_actions
        self.support = jnp.linspace(config.v_min, config.v_max, config.atoms)
        self.policy = DQNPolicy(n_channels, n_actions, rngs, config, support=self.support)

        self.model_out = nnx.jit(make_output_fn(config))
        self.loss_fn = make_loss_fn(config)
        self.targets_fn = nnx.jit(make_targets_fn(config, self.support))

        if config.ddqn:
            self.target_model = make_network(n_channels, n_actions, rngs, config)
            nnx.update(self.target_model, nnx.state(self.policy.model))

        self.optimizer = make_optim(self.policy.model, self.config)

    @staticmethod
    @nnx.jit(static_argnames=['loss_fn'])
    def train_step(model, optimizer, targets, states, actions, key, loss_fn):
        grads, aux = nnx.grad(loss_fn, has_aux=True)(model, states, actions, targets, key)
        optimizer.update(model, grads)  # in-place updates
        return aux['errors']

    def update(self, buffer, rngs):
        # batches_per_epoch = buffer.size // self.config.batch_size
        agent_data = {'errors': []}
        for _ in range(self.config.epochs):
            states, actions, rewards, next_states, discounts, idxs = buffer.sample_batch(self.config.batch_size, rngs())
            next_actions, _ = self.policy(next_states, rngs())
            target_model = self.target_model if self.config.ddqn else self.policy.model
            next_values = self.model_out(target_model, next_states, rngs())
            greedy_values = select_actions(next_values, next_actions)

            targets = self.targets_fn(rewards, discounts, greedy_values)
            errors = self.train_step(self.policy.model, self.optimizer, targets, states,
                                     actions, rngs(), self.loss_fn)
            # Priority replay
            buffer.update_priorities(idxs, errors)

            agent_data['errors'].append(jnp.mean(jnp.abs(errors)))

            if self.config.ddqn:
                new_target_params = optax.incremental_update(nnx.state(self.policy.model), nnx.state(self.target_model),
                                                             self.config.polyak_tau)
                nnx.update(self.target_model, new_target_params)

        return agent_data

    def behavioral_policy(self):
        if self.config.noisy_nets:
            return self.policy
        else:
            return EpsilonGreedyPolicy(self.policy, self.n_actions, self.config.epsilon)