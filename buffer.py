from abc import abstractmethod

from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class ReplayBuffer:
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    priorities: jnp.ndarray
    ptr: int = 0
    size: int = 0
    max_size: int = 10000
    omega: float = 0.6

    @classmethod
    def create(cls, obs_shape, config):
        max_size = config.buffer_size
        buffer = ReplayBuffer(
            obs=jnp.zeros((max_size, *obs_shape), dtype=jnp.float32),
            actions=jnp.zeros((max_size,), dtype=jnp.int32),
            rewards=jnp.zeros((max_size,), dtype=jnp.float32),
            next_obs=jnp.zeros((max_size, *obs_shape), dtype=jnp.float32),
            dones=jnp.zeros((max_size,), dtype=jnp.int32),
            priorities=jnp.zeros((max_size,), dtype=jnp.float32),
            ptr=0,
            size=0,
            max_size=max_size,
            omega=config.omega
        )
        return buffer

    def add_data(self, obs, actions, rewards, next_obs, dones, _):
        obs = obs.reshape(-1, *obs.shape[2:])
        actions = actions.reshape(-1)
        rewards = rewards.reshape(-1)
        next_obs = next_obs.reshape(-1, *next_obs.shape[2:])
        dones = dones.reshape(-1)

        batch_size = obs.shape[0]
        start = self.ptr
        end = self.ptr + batch_size
        idxs = jnp.arange(start, end) % self.max_size
        buffer = ReplayBuffer(
            obs=self.obs.at[idxs].set(obs),
            actions=self.actions.at[idxs].set(actions),
            rewards=self.rewards.at[idxs].set(rewards),
            next_obs=self.next_obs.at[idxs].set(next_obs),
            dones=self.dones.at[idxs].set(dones),
            priorities=self.priorities.at[idxs].set(jnp.ones_like(rewards)),
            ptr=end % self.max_size,
            size=jnp.minimum(self.size + batch_size, self.max_size),
            max_size=self.max_size,
            omega=self.omega
        )
        return buffer

    def sample_batch(self, batch_size, key):
        probs = self.priorities ** self.omega
        probs = probs / probs.sum()
        idxs = jax.random.choice(key, len(probs), (batch_size,), p=probs)

        return self.obs[idxs], self.actions[idxs], self.rewards[idxs], self.next_obs[idxs], self.dones[idxs], idxs

    def update_priorities(self, idxs, priorities):
        return self.replace(priorities=self.priorities.at[idxs].set(priorities))

@struct.dataclass
class RolloutBuffer:
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray

    @classmethod
    def create(cls, obs_shape, max_size=1):
        buffer = RolloutBuffer(
            obs=jnp.empty((max_size, *obs_shape), dtype=jnp.float32),
            actions=jnp.zeros((max_size,), dtype=jnp.int32),
            rewards=jnp.zeros((max_size,), dtype=jnp.float32),
            dones=jnp.zeros((max_size,), dtype=jnp.float32),
            log_probs=jnp.zeros((max_size, ), dtype=jnp.float32),
            values=jnp.zeros((max_size,), dtype=jnp.float32),
        )
        return buffer

    def add_data(self, obs, actions, rewards, _, dones, policy_aux):
        buffer = RolloutBuffer(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=policy_aux['log_probs'],
            values=policy_aux['values'],
        )
        return buffer

    def get(self):
        return self.obs, self.actions, self.rewards, self.dones, self.log_probs, self.values

def store(storage, step: slice | int, **kwargs):
    replace = {key: getattr(storage, key).at[step].set(value) for key, value in kwargs.items()}
    storage = storage.replace(**replace)
    return storage


def make_buffer(config, obs_shape):
    if config.strategy == 'dqn':
        return ReplayBuffer.create(obs_shape, config)
    elif config.strategy == 'ppo':
        return RolloutBuffer.create(obs_shape)
    else:
        raise NotImplementedError(f'{config.strategy} does not have a buffer')

