import gymnax
from flax import struct
import jax.numpy as jnp
from jax import vmap
import jax

@struct.dataclass
class EnvState:
    obs: jnp.ndarray
    state: jax.Array

class BatchEnv:
    def __init__(self, env, config):
        self.env = env
        self.num_envs = config.num_envs

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