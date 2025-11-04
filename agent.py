from abc import abstractmethod, ABC

import distrax
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.core import FrozenDict

from network_utils import network_init, network_forward, train_step
from neural_networks import CNN
from rl_helpers import generalized_advantage_estimation, q_loss_fn, ppo_loss_fn


class Agent(ABC):
    def __init__(self, env, env_params, config):
        self.n_actions = env.action_space(env_params).n
        self.n_states = env.observation_space(env_params).shape
        self.config = config

    @abstractmethod
    def init_state(self, key):
        pass

    @abstractmethod
    def select_action(self, obs, agent_state, key):
        pass

    @abstractmethod
    def update(self, state, buffer, final_obs, key):
        pass



class RandomAgent(Agent):
    def init_state(self, _):
        return None

    def select_action(self, obs, _, key):
        return jax.random.randint(key, shape=self.config.num_envs,
                                  minval=0, maxval=self.n_actions), {}

    def update(self, *args, **kwargs):
        return None


class DQNAgent(Agent):
    def __init__(self, env, env_params, config):
        super().__init__(env, env_params, config)
        self.network = CNN(self.n_actions)

    class DQNTrainState(train_state.TrainState):
        target_params: jax.FrozenDict

    def init_state(self, key):
        params, tx = network_init(key, self.network, self.n_states, self.config)
        state = self.DQNTrainState.create(apply_fn=self.network.apply, params=params, tx=tx,
                                          target_params=params)
        return state

    def select_action(self, obs, agent_state, *args, **kwargs):
        action = jnp.argmax(network_forward(agent_state, obs), axis=-1)
        return action, {}

    def update(self, agent_state, buffer, _, key):
        for _ in range(self.config.n_epochs):
            states, actions, rewards, next_states, dones = buffer.sample_batch(self.config.batch_size, key)
            q_values_next = agent_state.apply_fn(agent_state.target_params, next_states)
            targets = rewards + self.config.gamma * (1.0 - dones) * jnp.max(q_values_next, axis=-1)

            agent_state = train_step(agent_state, q_loss_fn, targets, states, actions)

            new_target_params = optax.incremental_update(agent_state.params, agent_state.target_params,
                                                         self.config.polyak_tau)
            agent_state = agent_state.replace(target_params=new_target_params)
        return agent_state

class PPOAgent(Agent):
    def __init__(self, env, env_params, config):
        super().__init__(env, env_params, config)
        self.network = CNN(self.n_actions + 1)

    def init_state(self, key):
        params, tx = network_init(key, self.network, self.n_states, self.config)
        state = train_state.TrainState.create(apply_fn=self.network.apply, params=params, tx=tx)
        return state

    def select_action(self, obs, agent_state, key, **kwargs):
        result = network_forward(agent_state, obs)
        action_logits = result[:, :-1]
        policy = distrax.Categorical(logits=action_logits)
        action = policy.sample(seed=key)
        log_probs = policy.log_prob(action).squeeze()

        return action, {'log_probs': log_probs, 'values': result[:, -1]}

    def update(self, agent_state, buffer, final_obs, key):
        obs, actions, rewards, dones, log_probs, values = buffer.get()
        final_value = network_forward(agent_state, final_obs)[:, -1]

        advantages = generalized_advantage_estimation(values, rewards, dones, final_value,
                                                      self.config.gamma, self.config.gae_lambda)
        advantages = advantages.reshape(-1)
        returns = advantages + values.reshape(-1)
        obs = jnp.reshape(obs, (-1, *obs.shape[2:]))
        actions = jnp.reshape(actions, -1)
        log_probs = jnp.reshape(log_probs, -1)
        epoch_size = obs.shape[0]
        for _ in range(self.config.n_epochs):
            epoch_indices = jax.random.permutation(key, epoch_size, independent=True)
            for start_idx in range(0, epoch_size, self.config.batch_size):
                end_idx = start_idx + self.config.batch_size
                idxs = epoch_indices[start_idx:end_idx]
                agent_state = train_step(agent_state, ppo_loss_fn, (returns[idxs], advantages[idxs]),
                                         obs[idxs], actions[idxs], log_probs[idxs],
                                          self.config.clip_threshold, self.config.entorpy_weight,
                                          self.config.value_weight)
        return agent_state

def make_agent(strategy, env, env_params, config, key):
    if strategy == 'dqn':
        agent = DQNAgent(env, env_params, config)
    elif strategy == 'ppo':
        agent = PPOAgent(env, env_params, config)
    elif strategy == 'random':
        agent = RandomAgent(env, env_params, config)
    else:
        raise NotImplementedError(f'Unknown strategy {strategy}')
    return agent, agent.init_state(key)
