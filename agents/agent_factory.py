from agents.agent import Agent
import jax

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent


class RandomAgent(Agent):
    def select_action(self, obs, key):
        return jax.random.randint(key, shape=obs.shape[0],
                                  minval=0, maxval=self.n_actions), {}

    def update(self, *args, **kwargs):
        return None

def make_agent(strategy, env, rngs, config):
    if strategy == 'dqn':
        agent = DQNAgent(env, rngs, config)
    elif strategy == 'ppo':
        agent = PPOAgent(env, rngs, config)
    elif strategy == 'random':
        agent = RandomAgent(env, rngs, config)
    else:
        raise NotImplementedError(f'Unknown strategy {strategy}')
    return agent
