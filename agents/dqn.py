import optax
from flax import nnx
from flax.training import train_state
from flax.core import FrozenDict
import jax.numpy as jnp
import jax

from agents.agent import Agent
from neural_networks import ActorCritic, OldCNN


class DQNAgent(Agent):
    def __init__(self, obs_shape, n_actions, rngs, config):
        super().__init__(obs_shape, n_actions, rngs, config)
        self.model = OldCNN(obs_shape[-1], n_actions, rngs, config)
        self.target_model = OldCNN(obs_shape[-1], n_actions, rngs, config)
        self.optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate=self.config.learning_rate), wrt=nnx.Param)

    class DQNTrainState(train_state.TrainState):
        target_params: FrozenDict

    @staticmethod
    def get_q_values(model, observations):
        return model(observations)
        # advantages, values = model(observations)
        # q_values = values + (advantages - advantages.mean())
        # return q_values

    def select_action(self, observations, *args, **kwargs):
        q_values = self.get_q_values(self.model, observations)
        print('selecting_action')
        action = jnp.argmax(q_values, axis=-1)
        return action, {}

    @staticmethod
    @nnx.jit
    def train_step(model, optimizer, targets, states, actions):
        def loss_fn(model):
            q_values = DQNAgent.get_q_values(model, states)
            q_selected = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)
            td_errors = optax.squared_error(q_selected, targets)
            loss = jnp.mean(td_errors)
            return loss, {'td_errors': td_errors}

        grads, aux = nnx.grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)  # in-place updates
        return aux['td_errors']

    def update(self, buffer, rngs):
        # batches_per_epoch = buffer.size // self.config.batch_size
        for _ in range(self.config.n_epochs):
            states, actions, rewards, next_states, dones, idxs = buffer.sample_batch(self.config.batch_size, rngs())

            q_values_next = nnx.jit(self.get_q_values)(self.target_model, next_states)
            targets = rewards + self.config.gamma * (1.0 - dones) * jnp.max(q_values_next, axis=-1)

            td_errors = self.train_step(self.model, self.optimizer, targets, states, actions)

            # DDQN
            new_target_params = optax.incremental_update(nnx.state(self.model), nnx.state(self.target_model),
                                                         self.config.polyak_tau)
            nnx.update(self.target_model, new_target_params)

            # Priority replay
            buffer.update_priorities(idxs, jnp.abs(td_errors))