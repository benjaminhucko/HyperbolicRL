import jax
from flax import nnx

from agents.agent_factory import make_agent
from buffer import make_buffer
from config import get_config
from environment import make_env
from logger import Logger

def run_episode_(agent, init_env, first_obs, n_steps: int, key):
    def run_step(state, keys):
        prev_obs, env = state
        agent_key, env_key = keys
        action, aux = agent.select_action(prev_obs, agent_key)
        next_obs, reward, dones = env.step(action, env_key)
        return (next_obs, env), (prev_obs, action, reward, next_obs, dones, aux)

    keys = jax.random.split(key, (n_steps, 2))
    end_state, data = jax.lax.scan(run_step, (first_obs, init_env), keys)
    return end_state[0], data

run_episode = nnx.jit(run_episode_, static_argnames=('n_steps'))

def sample_init_data(env, first_obs, rngs, config):
    random_agent = make_agent('random', env, rngs, config)
    next_state, data = run_episode(random_agent, env, first_obs, config.update_after, rngs())
    return next_state, data

def train_agent(env, config):
    logger = Logger(config)
    rngs = nnx.Rngs(config.seed)
    obs_shape, n_actions = env.observation_space().shape, env.action_space().n
    agent = make_agent(config.strategy, obs_shape, n_actions, rngs, config)
    buffer = make_buffer(config, obs_shape)
    next_obs = env.reset(rngs())
    if config.sample_init:
        next_obs, data = sample_init_data(env, next_obs, rngs, config)
        buffer = buffer.add_data(*data)

    for update_idx in range(config.num_updates):
        next_obs, data = run_episode(agent, env, next_obs, config.update_every, rngs())
        buffer = buffer.add_data(*data)
        agent.update(buffer, rngs)
        logger.log(data)
    return agent

def main():
    config = get_config()

    env = make_env(config.env_name, config.num_envs)
    agent = train_agent(env, config)

if __name__ == '__main__':
    main()