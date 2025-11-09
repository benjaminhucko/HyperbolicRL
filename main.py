import gymnax
import jax
import jax.numpy as jnp
from flax import nnx
from jax import vmap
from optax.schedules import piecewise_interpolate_schedule

from agents.agent_factory import make_agent
from buffer import ReplayBuffer, make_buffer
from config import get_config
from environment import EnvState, BatchEnv
from logger import Logger


def run_episode_(key, agent, env, env_params, env_state: EnvState, n_steps: int):
    keys = jax.random.split(key, n_steps)
    def run_step(last_env_state: EnvState, key):
        agent_key, env_key, reset_key = jax.random.split(key, 3)
        action, aux = agent.select_action(last_env_state.obs, key=agent_key)
        obs, env_hidden, reward, dones, _ = env.step(env_key, last_env_state.state, action, env_params)

        obs, env_hidden = env.reset_done(dones, key, env_params, obs, env_hidden)
        next_state = EnvState(obs=obs, state=env_hidden)
        return next_state, (last_env_state.obs, action, reward, next_state.obs, dones, aux)

    end_state, data = jax.lax.scan(run_step, env_state, keys)
    return end_state, data

run_episode = nnx.jit(run_episode_, static_argnames=['env', 'n_steps'])

def sample_init_data(burn_in_key, env, env_params, env_init_state: EnvState, config):
    obs_shape, n_actions = env.observation_space(env_params).shape, env.action_space(env_params).n
    random_agent = make_agent('random', obs_shape, n_actions, None, config)
    next_state, data = run_episode(burn_in_key, random_agent, env, env_params,
                                   env_state=env_init_state,
                                   n_steps=config.update_after)
    return next_state, data

def train_agent(env, env_params, config):
    logger = Logger(config)

    key = jax.random.PRNGKey(config.seed)
    key, agent_init_key, env_init_key, burn_in_key = jax.random.split(key, 4)

    obs_shape, n_actions = env.observation_space(env_params).shape, env.action_space(env_params).n
    print(obs_shape, n_actions)
    rngs = nnx.Rngs(config.seed)
    agent = make_agent(config.strategy, obs_shape, n_actions, rngs, config)
    buffer = make_buffer(config, obs_shape)

    env = BatchEnv(env, config.num_envs)
    obs, env_state = env.reset(env_init_key, env_params)
    next_state = EnvState(obs=obs, state=env_state)

    if config.strategy == 'dqn':
        next_state, data = sample_init_data(burn_in_key, env, env_params, next_state, config)
        buffer = buffer.add_data(*data)

    update_keys = jax.random.split(key, config.num_updates)
    for update_idx in range(config.num_updates):
        episode_key, update_key = jax.random.split(update_keys[update_idx], 2)
        next_state, data = run_episode(episode_key, agent, env, env_params,
                                       next_state, config.update_every)
        buffer = buffer.add_data(*data)
        agent.update(buffer, rngs)
        logger.log(data)
    return agent

def main():
    config = get_config()
    env, env_params = gymnax.make(config.env_name)
    agent = train_agent(env, env_params, config)

if __name__ == '__main__':
    main()