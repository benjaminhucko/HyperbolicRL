import time

from jax import vmap
from tensorboardX import SummaryWriter
import jax.numpy as jnp

class Logger:
    def __init__(self, config):
        self.writer = SummaryWriter(logdir=f"./logs/{config.strategy}_{int(time.time())}")
        self.cached_episode_length = jnp.zeros(config.num_envs)
        self.cached_episode_return = jnp.zeros(config.num_envs)
        self.config = config
        self.step = 0

        self.logged_keys = [
            "episode_length",
            "episode_return",
            "average_reward",
        ]

    def publish(self, logged_data):
        for key, value in logged_data.items():
            if len(value) == 0:
                continue
            average_value = sum(value) / len(value)
            self.writer.add_scalar(f"train/{key}", average_value, self.step)
        self.writer.flush()

    def log(self, data):
        self.step += 1
        logged_data = {key: [] for key in self.logged_keys}
        print(data[2].shape, data[4].shape)
        rewards = data[2].T
        dones = data[4].T
        for env_idx in range(self.config.num_envs):
            dones_indices = jnp.where(dones[env_idx])[0]
            episode_rewards = jnp.split(rewards[env_idx], dones_indices)

            env_lengths = [len(rewards) for rewards in episode_rewards]
            env_lengths[0] += self.cached_episode_length[env_idx]
            self.cached_episode_length.at[env_idx].set(env_lengths[-1])
            logged_data['episode_length'].extend(env_lengths[:-1])

            env_returns = [jnp.sum(rewards) for rewards in episode_rewards]
            env_returns[0] += self.cached_episode_return[env_idx]
            self.cached_episode_return.at[env_idx].set(env_returns[-1])
            logged_data['episode_return'].extend(env_returns[:-1])
        logged_data['average_reward'] = [jnp.mean(rewards)]
        self.publish(logged_data)