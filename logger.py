import time

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
        for env_idx in range(self.config.num_envs):
            rewards = data[2][env_idx]
            dones = data[4][env_idx]

            cached_episode_length = self.cached_episode_length[env_idx]
            cached_episode_return = self.cached_episode_return[env_idx]
            if not dones.any():
                continue

            done_indices = jnp.where(dones)[0]
            episode_lengths = jnp.diff(jnp.concatenate([jnp.array([-1]), done_indices + 1]))
            episode_lengths = episode_lengths.at[0].add(cached_episode_length)

            start_idx = 0
            for episode_length in episode_lengths:
                logged_data["episode_length"].append(episode_length.item())
                episode_return = cached_episode_return + jnp.sum(rewards[start_idx:start_idx + episode_length])
                logged_data['episode_return'].append(episode_return.item())
                cached_episode_return = 0
                average_reward = episode_return / episode_length
                logged_data["average_reward"].append(average_reward.item())

            cached_episode_length = len(dones) - done_indices[-1] + 1
            self.cached_episode_length = self.cached_episode_length.at[env_idx].set(cached_episode_length)
            cached_episode_return = jnp.sum(rewards[done_indices[-1] + 1:])
            self.cached_episode_return = self.cached_episode_return.at[env_idx].set(cached_episode_return)
        self.publish(logged_data)
