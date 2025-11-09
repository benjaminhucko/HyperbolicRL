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
    parser.add_argument('--polyak-tau', type=float, default=5e-4)
    parser.add_argument('--buffer-size', type=int, default=10000)
    parser.add_argument('--omega', type=float, default=0.6)

    # Convergence args
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--n-epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--activation', type=str, default='relu')
    ## CNN
    parser.add_argument('--hidden-channels', type=int, default=64)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)

    ## MLP
    parser.add_argument('--hidden-features', type=int, default=25)


    return parser.parse_args()

def get_config():
    config = parse_args()
    config.sample_init = (config.strategy == 'dqn')
    return config