from abc import abstractmethod, ABC

from flax import nnx

class Agent(ABC, nnx.Module):
    def __init__(self, obs_shape, n_actions, rng, config):
        self.n_states = nnx.static(obs_shape)
        self.n_actions = nnx.static(n_actions)
        self.config = nnx.static(config)

    @abstractmethod
    def update(self, buffer, key):
        pass
