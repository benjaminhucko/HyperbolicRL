from agents.agent import Agent
import jax

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent


class RandomAgent(Agent):
    def select_action(self, obs, key):
        print(self.n_actions)
        return jax.random.randint(key, shape=obs.shape[0],
                                  minval=0, maxval=self.n_actions), {}

    def update(self, *args, **kwargs):
        return None

def make_agent(strategy, obs_shape, n_actions, rngs, config):
    if strategy == 'dqn':
        agent = DQNAgent(obs_shape, n_actions, rngs, config)
    elif strategy == 'ppo':
        agent = PPOAgent(obs_shape, n_actions, rngs, config)
    elif strategy == 'random':
        agent = RandomAgent(obs_shape, n_actions, rngs, config)
    else:
        raise NotImplementedError(f'Unknown strategy {strategy}')
    return agent
