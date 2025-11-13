import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # Setup args
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env-name', type=str, default='Breakout-MinAtar')
    parser.add_argument('--strategy', type=str, default='dqn')
    parser.add_argument('--num-envs', type=int, default=16)

    # Update frequency args
    parser.add_argument('--update-after', type=int, default=100)
    parser.add_argument('--update-every', type=int, default=100)
    parser.add_argument('--num-updates', type=int, default=300)

    # PPO args
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--value-weight', type=float, default=0.5)
    parser.add_argument('--clip-threshold', type=float, default=0.1)
    parser.add_argument('--entorpy-weight', type=float, default=0.01)

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
    parser.add_argument('--std_init', type=float, default=0.5) # noisy nets init
    parser.add_argument('--atoms', type=int, default=10)
    parser.add_argument('--v-min', type=float, default=0.0)
    parser.add_argument('--v-max', type=float, default=5.0)

    # Convergence args
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--epsilon', type=float, default=0.2)

    ## CNN
    parser.add_argument('--hidden-channels', type=int, default=64)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)

    ## MLP
    parser.add_argument('--hidden-features', type=int, default=256)
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
    if not config.distributional:
        config.atoms = 1
    return config

def get_config():
    config = parse_args()
    config = apply_rainbow_flags(config)
    config.sample_init = (config.strategy == 'dqn')

    return config