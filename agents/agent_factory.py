from agents.agent import Agent
import jax

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent

def make_agent(strategy, obs_shape, n_actions, rngs, config):
    if strategy == 'dqn':
        agent = DQNAgent(obs_shape, n_actions, rngs, config)
    elif strategy == 'ppo':
        agent = PPOAgent(obs_shape, n_actions, rngs, config)
    else:
        raise NotImplementedError(f'Unknown strategy {strategy}')
    return agent
