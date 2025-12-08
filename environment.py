import time
from datetime import datetime
from functools import partial
from pathlib import Path

import gymnax
import jax
import jax.numpy as jnp
from flax import struct, nnx
from jax import vmap
from gymnax.visualize import Visualizer


@struct.dataclass
class EnvState:
    obs: jnp.ndarray
    state: jax.Array

class BatchEnv:
    def __init__(self, env, num_envs):
        self.env = env
        self.num_envs = num_envs

    def action_space(self, params):
        return self.env.action_space(params)

    def observation_space(self, params):
        return self.env.observation_space(params)

    def _batch_keys(self, keys):
        return jax.random.split(keys, self.num_envs)

    def reset_done(self, dones, key, env_params, b_obs, b_env_hidden):
        def reset_done_single(done, reset_key, obs, env_hidden):
            return jax.lax.cond(
                done,
                lambda _: self.env.reset(reset_key, env_params),
                lambda _: (obs, env_hidden),
                operand=None
            )
        keys = self._batch_keys(key)
        return vmap(reset_done_single)(dones, keys, b_obs, b_env_hidden)

    def reset(self, key, *args):
        batch_keys = self._batch_keys(key)
        return vmap(self.env.reset, in_axes=(0, None))(batch_keys, *args)

    def step(self, key, *args):
        batch_keys = self._batch_keys(key)
        return vmap(self.env.step, in_axes=(0, 0, 0, None))(batch_keys, *args)

class XEnvironment(nnx.Module):
    def __init__(self, env, env_params, obs_shape):
        self.env = nnx.static(env)
        self.env_params = nnx.static(env_params)
        self.env_state = nnx.data(None)
        self.obs_shape = self.env.observation_space(self.env_params).shape

        if obs_shape == 'channel_first':
            self.obs_fn = partial(jnp.transpose, axes=(0, 3, 1, 2))
            self.obs_shape = (self.obs_shape[2], self.obs_shape[0], self.obs_shape[1])
        else:
            self.obs_fn = nnx.identity

    def action_space(self):
        return self.env.action_space(self.env_params)

    def observation_shape(self):
        return self.obs_shape

    def reset(self, key):
        obs, state = self.env.reset(key, self.env_params)
        self.env_state = nnx.data(state)
        obs = self.obs_fn(obs)
        return obs

    def step(self, action, key):
        obs, state, reward, done, _ = self.env.step(key, self.env_state, action, self.env_params)
        self.env_state = nnx.data(state)
        obs = self.obs_fn(obs)
        return obs, reward, done

def env_name_factory(env_name):
    match env_name:
        case 'breakout': return 'Breakout-MinAtar'
        case 'asterix': return 'Asterix-MinAtar'
        case 'freeway': return 'Freeway-MinAtar'
        case 'space_invaders': return 'SpaceInvaders-MinAtar'
        case 'seaquest': return 'Seaquest-MinAtar'

def make_env(env_name, num_envs=1, obs_shape='channel_last'):
    env_name = env_name_factory(env_name)
    env, env_params = gymnax.make(env_name)
    if num_envs > 1:
        env = BatchEnv(env, num_envs)
    return XEnvironment(env, env_params, obs_shape)

def visualize_performance(env_name, policy, key, obs_shape, config):
    xenv = make_env(env_name, obs_shape=obs_shape)
    env, env_params = xenv.env, xenv.env_params
    state_seq, reward_seq = [], []
    key, key_reset = jax.random.split(key)
    obs, env_state = env.reset(key_reset, env_params)
    while True:
        state_seq.append(env_state)
        key, key_act, key_step = jax.random.split(key, 3)

        obs = xenv.obs_fn(obs[None, :])
        action, _ = policy(obs, key_act)
        next_obs, next_env_state, reward, done, info = env.step(
            key_step, env_state, action.squeeze(), env_params
        )
        reward_seq.append(reward)
        if done:
            break
        else:
          obs = next_obs
          env_state = next_env_state

    cum_rewards = jnp.cumsum(jnp.array(reward_seq))
    vis = Visualizer(env, env_params, state_seq, cum_rewards)

    specification = 'base'
    if config.hyper:
        specification = 'hyper'
    elif config.hyperpp:
        specification = 'hyperpp'


    out_file_name = f'visualization/{config.geometry}/{config.strategy}/{specification}'
    Path(out_file_name).mkdir(parents=True, exist_ok=True)
    ts = time.time()
    dt = datetime.fromtimestamp(ts)

    formatted = dt.strftime("%m_%d_%H_%M")
    vis.animate(f"{out_file_name}/{formatted}.gif")