from agents.agent import Agent
import jax

from agents.dqn import DQNAgent
from agents.ppo import PPOAgent

def make_agent(strategy, n_channels, n_actions, rngs, config):
    if strategy == 'dqn':
        agent = DQNAgent(n_channels, n_actions, rngs, config)
    elif strategy == 'ppo':
        agent = PPOAgent(n_channels, n_actions, rngs, config)
    else:
        raise NotImplementedError(f'Unknown strategy {strategy}')
    return agent
