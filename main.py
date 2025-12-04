import time

import jax
from flax import nnx
from tqdm import tqdm

from agents.agent_factory import make_agent
from agents.random_policy import RandomPolicy
from buffer import make_buffer
from config import get_config
from environment import make_env, visualize_performance
from logger import Logger

@nnx.jit(static_argnames=['n_steps'])
def run_episode(policy, init_env, first_obs, n_steps: int, key):
    def run_step(state, keys):
        prev_obs, env = state
        agent_key, env_key = keys
        action, aux = policy(prev_obs, agent_key)
        next_obs, reward, dones = env.step(action, env_key)
        return (next_obs, env), (prev_obs, action, reward, next_obs, dones, aux)

    keys = jax.random.split(key, (n_steps, 2))
    end_state, data = jax.lax.scan(run_step, (first_obs, init_env), keys)
    return end_state[0], data

def sample_init_data(burn_in_key, env, env_init_obs, config):
    next_state, data = run_episode(RandomPolicy(env.action_space().n), env, first_obs=env_init_obs,
                                   n_steps=config.update_after, key=burn_in_key)
    return next_state, data

def train_agent(env, config):
    logger = Logger(config)
    rngs = nnx.Rngs(config.seed)
    obs_shape, n_actions = env.observation_space().shape, env.action_space().n
    agent = make_agent(config.strategy, obs_shape, n_actions, rngs, config)
    buffer = make_buffer(config, obs_shape)
    next_obs = env.reset(rngs())
    if config.sample_init:
        next_state, data = sample_init_data(rngs(), env, next_obs, config)
        buffer = buffer.add_data(*data)

    for update_idx in tqdm(range(config.num_updates)):
        next_obs, data = run_episode(agent.behavioral_policy(), env, next_obs, config.update_every, rngs())
        buffer = buffer.add_data(*data, next_obs)
        stats = agent.update(buffer, rngs)
        logger.log(data, stats)
    return agent

def main():
    config = get_config()
    obs_shape = 'channel_first' if config.geometry == 'hyperbolic' else 'channel_last'
    env = make_env(config.env_name, config.num_envs, obs_shape)
    agent = train_agent(env, config)
    visualize_performance(config.env_name, agent.policy, jax.random.PRNGKey(config.seed), obs_shape, config)

if __name__ == '__main__':
    main()