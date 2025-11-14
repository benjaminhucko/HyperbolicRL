import jax
from flax import nnx
import jax.numpy as jnp

class RandomPolicy(nnx.Module):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def __call__(self, obs, key):
        return jax.random.randint(key, shape=obs.shape[0], minval=0, maxval=self.n_actions), {}

class EpsilonGreedyPolicy(nnx.Module):
    def __init__(self, greedy_policy, n_actions, epsilon):
        self.n_actions = n_actions
        self.random_policy = RandomPolicy(n_actions)
        self.greedy_policy = greedy_policy
        self.epsilon = epsilon

    def __call__(self, obs, key):
        key1, key2, key3 = jax.random.split(key, 3)
        p = jax.random.uniform(key1, shape=obs.shape[0])
        action = jnp.where(p < self.epsilon,
                           self.random_policy(obs, key2)[0],
                           self.greedy_policy(obs, key3)[0])

        return action, {}


