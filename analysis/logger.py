import time

import jax
import math
from jax import vmap
from jedi.inference.gradual.base import BaseTypingValue
from tensorboardX import SummaryWriter
import jax.numpy as jnp

class Logger:
    def __init__(self, config):
        self.writer = SummaryWriter(logdir=f"./analysis/logs/{config.env}/{config.strategy}/{config.geometry}/"
                                           f"log_{int(time.time())}")
        self.cached_episode_length = jnp.zeros(config.num_envs)
        self.cached_episode_return = jnp.zeros(config.num_envs)
        self.config = config
        self.rollout = 0

        self.env_keys = [
            "episode_length",
            "episode_return",
            "average_reward",
        ]

    def publish(self, logged_data, step):
        for key, value in logged_data.items():
            self.writer.add_scalar(f"train/{key}", value, step)

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

        # THiS IS WRONG
        if jnp.any(batched_dones):
            logged_data['episode_length'] = jnp.sum(emitted_length) / jnp.sum(batched_dones)
            logged_data['episode_return'] = jnp.sum(emitted_return) / jnp.sum(batched_dones)
        logged_data['average_reward'] = jnp.mean(batched_rewards)
        self.publish(logged_data, self.get_env_interactions())

    def get_env_interactions(self):
        return self.config.num_envs * (self.rollout * self.config.update_every + self.config.update_after)

    def log_agent(self, data):
        for key in data.keys():
            data[key] = sum(data[key]) / len(data[key])
        self.publish(data, self.get_env_interactions())

    def log_agent_per_epoch(self, data):
        steps_in_epoch = math.ceil(self.config.update_every * self.config.num_envs / self.config.batch_size)
        for epoch in range(self.config.epochs):
            epoch_data = {}
            epoch_slice = slice(epoch * steps_in_epoch, (epoch + 1) * steps_in_epoch)
            for key in data.keys():
                per_batch_values = data[key][epoch_slice]
                epoch_data[key] = sum(per_batch_values) / len(per_batch_values)
            self.publish(epoch_data, epoch)

    def log(self, env_data, agent_data=None):
        if self.config.updates == 1:
            self.log_agent_per_epoch(agent_data)
        else:
            self.log_agent(agent_data)
            self.log_env(env_data)
        self.writer.flush()
        self.rollout += 1






