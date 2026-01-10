import json
import pprint

import jax
from flax import nnx
from tqdm import tqdm

from agents.agent_factory import make_agent
from agents.random_policy import RandomPolicy
from analysis.analyzer import Analyzer
from analysis.evaluator import Evaluator
from analysis.model_serializer import ModelSerializer
from buffer import make_buffer
from config import get_config
from environment import make_env, visualize_performance, StickyAction
from analysis.logger import Logger

import jax.numpy as jnp

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
    return end_state[1], end_state[0], data

def sample_init_data(burn_in_key, env, env_init_obs, config):
    next_state, data = run_episode(RandomPolicy(env.action_space().n), env, first_obs=env_init_obs,
                                   n_steps=config.update_after, key=burn_in_key)
    return next_state, data

def train_agent(env, config, rngs):
    logger = Logger(config)
    analyzer = Analyzer(config)
    with open(f'{config.logging_dir}/config.txt', 'w+') as f:
        pprint.pprint(vars(config), stream=f)

    obs_shape, n_actions = env.observation_shape(), env.action_space().n
    n_channels = obs_shape[-1] if config.geometry != 'hyperbolic' else obs_shape[0]
    agent = make_agent(config.strategy, n_channels, n_actions, rngs, config)
    buffer = make_buffer(config, obs_shape)
    next_obs = env.reset(rngs())
    if config.sample_init:
        next_state, data = sample_init_data(rngs(), env, next_obs, config)
        buffer = buffer.add_data(*data)

    for update_idx in tqdm(range(config.updates), desc='update'):
        env, next_obs, data = run_episode(agent.behavioral_policy(), env, next_obs, config.update_every, rngs())
        buffer = buffer.add_data(*data, next_obs)
        stats = agent.update(buffer, rngs, analyzer.analyze_grads())
        analyzer.step(stats)

        logger.log(data, stats)
    if config.analyze:
        analyzer.plot_all()

    return agent

def eval_agent(xenv, policy, rngs, config, sticky_sigma=0.0):
    if sticky_sigma > 0:
        env = StickyAction(xenv, sigma=sticky_sigma)
    else:
        env = xenv

    finished = jnp.zeros((config.eval_episodes,))
    returns = jnp.zeros((config.eval_episodes,))

    obs = env.reset(rngs())
    while not jnp.all(finished):
        action, aux = policy(obs, rngs())
        obs, reward, dones = env.step(action, rngs())
        finished = jnp.where(dones, True, finished)
        returns = jnp.where(finished, returns, returns + reward)
    return returns

def real_to_sim_gap_eval(env_name, policy, rngs, obs_shape, config):
    evaluator = Evaluator(config)
    policy = nnx.jit(policy)
    xenv = make_env(env_name, num_envs=config.eval_episodes, obs_shape=obs_shape)
    for sticky_chance in tqdm(range(0, 26, 1), desc='eval sticky chance'):
        sticky_sigma = sticky_chance / 100.0
        returns = eval_agent(xenv, policy, rngs, config, sticky_sigma)
        evaluator.log(sticky_sigma, returns)

def run_experiment(config):
    obs_format = 'channel_first' if config.geometry == 'hyperbolic' else 'channel_last'
    env = make_env(config.env, config.num_envs, obs_format)
    rngs = nnx.Rngs(config.seed)
    serializer = ModelSerializer(config)
    if not serializer.ckpt_exists():
        agent = train_agent(env, config, rngs)
        serializer.save(agent.policy.model)
    else:
        obs_shape, n_actions = env.observation_shape(), env.action_space().n
        n_channels = obs_shape[-1] if config.geometry != 'hyperbolic' else obs_shape[0]
        agent = make_agent(config.strategy, n_channels, n_actions, rngs, config)
        agent.policy.model = serializer.load(agent.policy.model)
    if config.eval:
        real_to_sim_gap_eval(config.env, agent.eval_policy(), rngs, obs_format, config)
    if config.visualize:
        visualize_performance(config.env, agent.eval_policy(), rngs(), obs_format, config)

def main():
    config = get_config()
    run_experiment(config)

if __name__ == '__main__':
    main()