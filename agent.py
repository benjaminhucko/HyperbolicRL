from abc import abstractmethod, ABC

import distrax
import jax.numpy as jnp
import jax
from jax import vmap

from network_utils import create_train_state, network_forward, train_step, q_loss_fn, ppo_loss_fn
from neural_networks import CNN

import rlax

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

    def init_state(self, key):
        state = create_train_state(key, self.network, self.n_states, self.config)
        return state

    def select_action(self, obs, agent_state, *args, **kwargs):
        action = jnp.argmax(network_forward(agent_state, obs), axis=-1)
        return action, {}

    def update(self, agent_state, buffer, _, key):
        for _ in range(self.config.n_epochs):
            states, actions, rewards, next_states, dones = buffer.sample_batch(self.config.batch_size, key)
            q_values_next = network_forward(agent_state, next_states)
            targets = rewards + self.config.gamma * (1.0 - dones) * jnp.max(q_values_next, axis=-1)

            agent_state = train_step(agent_state, q_loss_fn, targets, states, actions)
        return agent_state

class PPOAgent(Agent):
    def __init__(self, env, env_params, config):
        super().__init__(env, env_params, config)
        self.network = CNN(self.n_actions + 1)

    def init_state(self, key):
        state = create_train_state(key, self.network, self.n_states, self.config)
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

        # all_values = jnp.concatenate((values.squeeze(), final_value[None, :]), axis=0)
        # discounts = (1 - dones.squeeze()) * self.config.gamma
        print(jnp.unique(actions, return_counts=True))
        print(dones)
        advantages = self.generalized_advantage_estimation(values, rewards,
                                                           dones, final_value,
                                                           self.config.gamma, self.config.gae_lambda).reshape(-1)

        # batch_advantage_estimation = vmap(rlax.truncated_generalized_advantage_estimation,
        #                                   in_axes=(1, 1, None, 1))
        #
        # advantages = batch_advantage_estimation(rewards, discounts, self.config.gae_lambda, all_values)

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
                train_step(agent_state, ppo_loss_fn, (returns[idxs], advantages[idxs]),
                           obs[idxs], actions[idxs], log_probs[idxs],
                           self.config.clip_threshold, self.config.entorpy_weight,
                           self.config.value_weight)

        return agent_state

    @staticmethod
    @jax.jit
    def generalized_advantage_estimation(values, rewards, dones, term_value, discount_factor, lambda_):
        def fold_left(last_gae, rest):
            td_error, discount = rest
            last_gae = td_error + discount * lambda_ * last_gae
            return last_gae, last_gae

        discounts = jnp.where(dones, 0, discount_factor)
        td_errors = rewards + discounts * jnp.append(values[1:], jnp.expand_dims(term_value, 0), axis=0) - values

        _, advantages = jax.lax.scan(fold_left, jnp.zeros(td_errors.shape[1]), (td_errors, discounts), reverse=True)
        return advantages

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
