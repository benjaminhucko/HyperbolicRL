import time
from typing import Callable

import gymnax
import jax
from flax import nnx

from agents.agent_factory import make_agent
from agents.random_policy import RandomPolicy
from buffer import make_buffer
from config import get_config
from environment import EnvState, BatchEnv, visualize_performance
from logger import Logger

from tqdm import tqdm


@nnx.jit(static_argnames=['env', 'n_steps'])
def run_episode(key, policy: Callable, env, env_params, env_state: EnvState, n_steps: int):
    keys = jax.random.split(key, n_steps)
    def run_step(last_env_state: EnvState, key):
        agent_key, env_key, reset_key = jax.random.split(key, 3)
        action, aux = policy(last_env_state.obs, agent_key)
        obs, env_hidden, reward, dones, _ = env.step(env_key, last_env_state.state, action, env_params)

        obs, env_hidden = env.reset_done(dones, key, env_params, obs, env_hidden)
        next_state = EnvState(obs=obs, state=env_hidden)
        return next_state, (last_env_state.obs, action, reward, next_state.obs, dones, aux)

    end_state, data = jax.lax.scan(run_step, env_state, keys)
    return end_state, data

def sample_init_data(burn_in_key, env, env_params, env_init_state: EnvState, config):
    n_actions = env.action_space(env_params).n

    next_state, data = run_episode(burn_in_key, RandomPolicy(n_actions), env, env_params,
                                   env_state=env_init_state,
                                   n_steps=config.update_after)
    return next_state, data

def train_agent(env, env_params, config, rngs):
    logger = Logger(config)

    obs_shape, n_actions = env.observation_space(env_params).shape, env.action_space(env_params).n

    agent = make_agent(config.strategy, obs_shape, n_actions, rngs, config)
    buffer = make_buffer(config, obs_shape)

    env = BatchEnv(env, config.num_envs)
    obs, env_state = env.reset(rngs(), env_params)
    next_state = EnvState(obs=obs, state=env_state)

    if config.strategy == 'dqn':
        next_state, data = sample_init_data(rngs(), env, env_params, next_state, config)
        buffer = buffer.add_data(*data, next_state.obs)

    for update_idx in tqdm(range(config.num_updates)):
        start_time = time.time()
        next_state, data = run_episode(rngs(), agent.behavioral_policy(), env, env_params,
                                       next_state, config.update_every)
        print(f'Episode {update_idx} finished in {time.time() - start_time} seconds')
        start_time = time.time()
        buffer = buffer.add_data(*data, next_state.obs)
        print(f'data_added {update_idx} finished in {time.time() - start_time} seconds')
        start_time = time.time()
        agent.update(buffer, rngs)
        print(f'agent updated {update_idx} finished in {time.time() - start_time} seconds')

        logger.log(data)
    return agent

def main():
    config = get_config()
    env, env_params = gymnax.make(config.env_name)
    rngs = nnx.Rngs(config.seed)
    agent = train_agent(env, env_params, config, rngs)
    visualize_performance(env, env_params, agent.policy, rngs())

if __name__ == '__main__':
    main()