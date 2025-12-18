import copy
import json

from config import get_config
from main import run_experiment

SEEDS = [5]
# 20 epochs, 40 epochs, 30 epochs, 40 epochs -> hyperbolic
# 10 epochs, 10 epochs, 10 epochs, 10 epochs -> euclidean
# euclidean asterix sus

# ENVS = ["asterix", "breakout", "freeway", "space_invaders"]
ENVS = ["freeway", "asterix", "breakout", "space_invaders"]
ARCHITECTURES = ['euclidean', 'hyperbolic']
EXPERIMENTS = ['exp1']

# increase sigma
# go from 128 to 64 hidden

# orthogonal initialization


def set_env(defaults, env):
    defaults = copy.deepcopy(defaults)
    defaults['env'] = env
    if not isinstance(defaults['epochs'], int):
        defaults['epochs'] = defaults['epochs'][env]
    return defaults

def multiple_seed_experiment(defaults):
    for seed in SEEDS:
        for env in ENVS:
            env_defaults = set_env(defaults, env)
            env_defaults['seed'] = seed
            config = get_config(env_defaults)
            run_experiment(config)


def defaults_from_json(defaults_path):
    with open(defaults_path, 'r') as f:
        defaults = json.load(f)

    return defaults

def main():
    # curvature 0.01 -> not too bad. Asterix breaks. Otherwise competitive
    for arch in ARCHITECTURES:
        for experiment in EXPERIMENTS:
            experiment_file = f'experiments/{arch}/{experiment}.json'
            defaults = defaults_from_json(experiment_file)
            multiple_seed_experiment(defaults)

if __name__ == '__main__':
    main()