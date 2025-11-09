import gymnax
from flax import struct, nnx
import jax.numpy as jnp
from gymnax.wrappers import gym
from jax import vmap
import jax

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
        obs = vmap(self.env.reset, in_axes=(0, None))(batch_keys, *args)
        return obs

    def step(self, key, *args):
        batch_keys = self._batch_keys(key)
        return vmap(self.env.step, in_axes=(0, 0, 0, None))(batch_keys, *args)

class XEnvironment(nnx.Module):
    def __init__(self, env, env_params):
        self.env = nnx.static(env)
        self.env_params = nnx.static(env_params)
        self.env_state = nnx.data(None)

    def action_space(self):
        return self.env.action_space(self.env_params)

    def observation_space(self):
        return self.env.observation_space(self.env_params)

    def reset(self, key):
        obs, state = self.env.reset(key, self.env_params)
        self.env_state = nnx.data(state)
        return obs

    def step(self, action, key):
        obs, state, reward, done, _ = self.env.step(key, self.env_state, action, self.env_params)
        self.env_state = nnx.data(state)
        return obs, reward, done


def make_env(env_name, num_envs=1):
    env, env_params = gymnax.make(env_name)
    batch_env = BatchEnv(env, num_envs)
    return XEnvironment(batch_env, env_params)
