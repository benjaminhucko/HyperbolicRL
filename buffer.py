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
    discounts: jnp.ndarray
    priorities: jnp.ndarray
    ptr: int = 0
    size: int = 0
    max_size: int = 10000
    gamma: float = 0.99
    omega: float = 0.6
    n: int = 0

    @classmethod
    def create(cls, obs_shape, config):
        max_size = config.buffer_size
        buffer = ReplayBuffer(
            obs=jnp.zeros((max_size, *obs_shape), dtype=jnp.float32),
            actions=jnp.zeros((max_size,), dtype=jnp.int32),
            rewards=jnp.zeros((max_size,), dtype=jnp.float32),
            next_obs=jnp.zeros((max_size, *obs_shape), dtype=jnp.float32),
            discounts=jnp.zeros((max_size,), dtype=jnp.float32),
            priorities=jnp.zeros((max_size,), dtype=jnp.float32),
            ptr=0,
            size=0,
            max_size=max_size,
            gamma=config.gamma,
            omega=config.omega,
            n=config.n_steps,
        )
        return buffer

    def n_step(self, next_obs, rewards, dones):
        T, B = rewards.shape
        def aggregation_fn(carry, state):
            reward_queue, prev_m = carry
            reward, done = state

            m = jnp.where(done, 0, prev_m)
            next_m = jnp.minimum(m + 1, self.n)

            reward_queue = jnp.where(done, jnp.zeros_like(reward_queue), reward_queue)
            reward_queue = jnp.roll(reward_queue, 1, axis=0).at[0, :].set(reward)
            n_step_rewards = jnp.sum(reward_queue, axis=0)
            next_reward_queue = reward_queue * self.gamma

            return (next_reward_queue, next_m), (n_step_rewards, m != self.n)

        reward_queue_init = jnp.zeros((self.n, B), dtype=jnp.float32)
        m_init = jnp.full(B, self.n)
        _, (rewards, dones) = jax.lax.scan(aggregation_fn, (reward_queue_init, m_init),
                                                  (rewards, dones), reverse=True)
        discounts = jnp.full_like(rewards, self.gamma ** self.n)
        trunc_discounts = self.gamma ** jnp.arange(self.n, 0, -1)
        discounts.at[discounts.shape[0] - self.n:].set(trunc_discounts[:, None])
        discounts = jnp.where(dones, 0, discounts)
        next_obs_n = self.n - 1
        shifted_obs = jnp.roll(next_obs, -next_obs_n, axis=0)
        next_obs = shifted_obs.at[T - next_obs_n:].set(next_obs[None, -1])

        return next_obs, rewards, discounts

    def add_data(self, obs, actions, rewards, next_obs, dones, *args):
        obs = obs.reshape(-1, *obs.shape[2:])
        actions = actions.reshape(-1)
        # n-step DQN
        if self.n > 0:
            next_obs, rewards, discounts = self.n_step(next_obs, rewards, dones)
        else:
            discounts = jnp.where(dones, 0, self.gamma)

        next_obs = next_obs.reshape(-1, *next_obs.shape[2:])
        rewards = rewards.reshape(-1)
        discounts = discounts.reshape(-1)

        max_priorities = jnp.max(self.priorities)
        max_priorities = max_priorities if max_priorities > 0 else 1

        batch_size = obs.shape[0]
        start = self.ptr
        end = self.ptr + batch_size
        idxs = jnp.arange(start, end) % self.max_size
        buffer = self.replace(
            obs=self.obs.at[idxs].set(obs),
            actions=self.actions.at[idxs].set(actions),
            rewards=self.rewards.at[idxs].set(rewards),
            next_obs=self.next_obs.at[idxs].set(next_obs),
            discounts=self.discounts.at[idxs].set(discounts),
            priorities=self.priorities.at[idxs].set(jnp.full_like(rewards, max_priorities)),
            ptr=end % self.max_size,
            size=jnp.minimum(self.size + batch_size, self.max_size),
        )
        return buffer

    def sample_batch(self, batch_size, key):
        probs = self.priorities ** self.omega
        probs = probs / probs.sum()
        idxs = jax.random.choice(key, len(probs), (batch_size,), p=probs)

        return self.obs[idxs], self.actions[idxs], self.rewards[idxs], self.next_obs[idxs], self.discounts[idxs], idxs

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
    final_obs: jnp.ndarray

    @classmethod
    def create(cls, obs_shape, max_size=1):
        buffer = RolloutBuffer(
            obs=jnp.empty((max_size, *obs_shape), dtype=jnp.float32),
            actions=jnp.zeros((max_size,), dtype=jnp.int32),
            rewards=jnp.zeros((max_size,), dtype=jnp.float32),
            dones=jnp.zeros((max_size,), dtype=jnp.float32),
            log_probs=jnp.zeros((max_size, ), dtype=jnp.float32),
            values=jnp.zeros((max_size,), dtype=jnp.float32),
            final_obs=jnp.zeros((max_size,), dtype=jnp.float32),
        )
        return buffer

    def add_data(self, obs, actions, rewards, _, dones, policy_aux, final_obs):
        buffer = RolloutBuffer(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            log_probs=policy_aux['log_probs'],
            values=policy_aux['values'],
            final_obs=final_obs,
        )
        return buffer

    def get(self):
        return self.obs, self.actions, self.rewards, self.dones, self.log_probs, self.values, self.final_obs

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

