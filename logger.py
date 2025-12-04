import time

import jax
from jax import vmap
from jedi.inference.gradual.base import BaseTypingValue
from tensorboardX import SummaryWriter
import jax.numpy as jnp

class Logger:
    def __init__(self, config):
        self.writer = SummaryWriter(logdir=f"./logs/{config.strategy}_{int(time.time())}")
        self.cached_episode_length = jnp.zeros(config.num_envs)
        self.cached_episode_return = jnp.zeros(config.num_envs)
        self.config = config
        self.step = 0

        self.env_keys = [
            "episode_length",
            "episode_return",
            "average_reward",
        ]

    def publish(self, logged_data):
        for key, value in logged_data.items():
            self.writer.add_scalar(f"train/{key}", value, self.step)

    def log_env(self, data):
        logged_data = {key: 0 for key in self.env_keys}
        batched_rewards, batched_dones = data[2], data[4]

        def traverse_data(carry, state):
            episode_length, episode_return = carry
            rewards, dones = state

            episode_length += jnp.ones_like(rewards)
            episode_return += rewards

            emit_length = jnp.where(dones, episode_length, 0)
            emit_return = jnp.where(dones, episode_return, 0)

            episode_length = jnp.where(dones, 0, episode_length)
            episode_return = jnp.where(dones, 0, episode_return)
            return (episode_length, episode_return), (emit_length, emit_return)

        next_cache, emitted_data = jax.lax.scan(traverse_data, (self.cached_episode_length, self.cached_episode_return),
                                                (batched_rewards, batched_dones))

        self.cached_episode_length, self.cached_episode_return = next_cache
        emitted_length, emitted_return = emitted_data
        logged_data['episode_length'] = jnp.sum(emitted_length) / jnp.sum(batched_dones)
        logged_data['episode_return'] = jnp.sum(emitted_return) / jnp.sum(batched_dones)
        logged_data['average_reward'] = jnp.mean(batched_rewards)
        self.publish(logged_data)

    def log_agent(self, data):
        for key in data.keys():
            data[key] = sum(data[key]) / len(data[key])
        self.publish(data)

    def log(self, env_data, agent_data=None):
        self.step += 1
        self.log_env(env_data)
        self.log_agent(agent_data)
        self.writer.flush()





