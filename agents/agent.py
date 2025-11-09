from abc import abstractmethod, ABC

from flax import nnx

class Agent(ABC, nnx.Module):
    def __init__(self, env, rngs, config):
        self.n_actions = nnx.static(env.action_space().n)
        self.n_states = nnx.static(env.observation_space().shape)
        self.config = nnx.static(config)

    @abstractmethod
    def select_action(self, obs, key):
        pass

    @abstractmethod
    def update(self, buffer, key):
        pass
