import argparse
import jax.numpy as jnp

def parse_args(defaults):
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry', type=str, choices=['euclidean', 'hyperbolic', 'hybrid'],
                        default='euclidean')
    parser.add_argument('--learn-curvature', action='store_true')

    # Hyper papers replication
    parser.add_argument('--hyper', action='store_true')
    parser.add_argument('--hyperpp', action='store_true')
    parser.add_argument('--baseline', action='store_true')

    # Setup args
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='breakout',
                        choices=['asterix', 'breakout', 'freeway', 'space_invaders'])
    parser.add_argument('--strategy', type=str, default='ppo')
    parser.add_argument('--num-envs', type=int, default=8) # old: 32

    # Update frequency args
    parser.add_argument('--update-after', type=int, default=0)
    parser.add_argument('--update-every', type=int, default=128) # old 100
    parser.add_argument('--updates', type=int, default=100)

    # PPO args
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--value-weight', type=float, default=0.5)
    parser.add_argument('--clip-threshold', type=float, default=0.1)
    parser.add_argument('--entropy-weight', type=float, default=0.01)
    parser.add_argument('--gauss-sigma', type=float, default=1)

    # RAINBOW args
    parser.add_argument('--duelling', action='store_true')
    parser.add_argument('--priority', action='store_true')
    parser.add_argument('--ddqn', action='store_true')
    parser.add_argument('--noisy-nets', action='store_true') # not converging
    parser.add_argument('--n-td', action='store_true')
    parser.add_argument('--distributional', action='store_true') # not converging

    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--polyak-tau', type=float, default=5e-3) # DDQN
    parser.add_argument('--omega', type=float, default=0.6) # priority
    parser.add_argument('--n-steps', type=int, default=4) # n_step td
    parser.add_argument('--std-init', type=float, default=0.1) # noisy nets init


    parser.add_argument('--atoms', type=int, default=51)
    parser.add_argument('--v-min', type=float, default=-10)
    parser.add_argument('--v-max', type=float, default=10)
    parser.add_argument('--log-size', type=int, default=100)

    # Convergence args
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--float64', action='store_true')

    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--epsilon', type=float, default=0.2)

    ## CNN
    parser.add_argument('--hidden-channels', type=int, default=16)

    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--n-conv', type=int, default=2)

    ## MLP
    parser.add_argument('--hidden-features', type=int, default=128) # 16
    parser.add_argument('--n-linear', type=int, default=2)

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--check-distribution', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval-episodes', type=int, default=200)
    parser.add_argument('--experiment', type=str, default="")

    parser.add_argument('--curvature', type=float, default=0.01)


    parser.set_defaults(**defaults)


    return parser.parse_args()

def apply_rainbow_flags(config):
    if config.strategy == 'rainbow':
        config.strategy = 'dqn'
        config.priority = True # works
        config.duelling = True # works
        config.ddqn = True # works
        config.n_td = True # works
        config.noisy_nets = True # works
        config.distributional = True # implementing
    if not config.priority:
        config.omega = 0

    if not config.ddqn:
        config.polyak_tau = 1

    if not config.n_td:
        config.n_steps = 0

    if config.noisy_nets:
        config.epsilon = 0

    if config.distributional:
        config.categorical_actor = True
    return config


def apply_hyper_flags(config):
    if config.hyper or config.hyperpp:
        config.n_linear = 1
        config.geometry = 'hybrid'
        config.kernel_size = 3
        config.stride = 1
        config.hidden_features = 256
    return config


def get_config(defaults=None):
    if defaults is None:
        defaults = {}
    config = parse_args(defaults)
    config.categorical_actor = False
    config.sample_init = (config.strategy == 'dqn')

    if config.strategy in ['dqn', 'rainbow']:
        config = apply_rainbow_flags(config)
    else:
        config = apply_hyper_flags(config)

    if not config.distributional:
        config.atoms = 1

    config.dtype = jnp.float64 if config.float64 else jnp.float32

    if len(config.experiment) > 0:
        config.logging_dir = f'logs/{config.experiment}/{config.env}/{config.geometry}/{config.seed}'
    else:
        config.logging_dir = f'logs/{config.env}/{config.geometry}/{config.seed}'
    return config