import copy
import json

import jax
from tqdm import tqdm

from config import get_config
from main import run_experiment

SEEDS = [1, 2, 3, 4, 5, 6]
# Asterix, Breakout, Freeway, Space_invaders
# 20 epochs, 40 epochs, 30 epochs, 40 epochs -> hyperbolic
# 15 epochs, 30 epochs, 20 epochs, 30 epochs -> old hyperbolic

# 10 epochs, 10 epochs, 10 epochs, 10 epochs -> euclidean
# euclidean asterix sus

ENVS = ['asterix']
ARCHITECTURES = ['euclidean', 'hyperbolic']
EXPERIMENTS = ['exp_main']

# increase sigma -> not help
# go from 128 to 64 hidden
jax.config.update("jax_enable_x64", True)

# orthogonal initialization

def set_env(defaults, env):
    defaults = copy.deepcopy(defaults)
    defaults['env'] = env
    if not isinstance(defaults['epochs'], int):
        defaults['epochs'] = defaults['epochs'][env]

    if 'curvature' in defaults and not isinstance(defaults['curvature'], float):
        defaults['curvature'] = defaults['curvature'][env]

    return defaults

def multiple_seed_experiment(defaults):
    for seed in tqdm(SEEDS, desc='seeds'):
        for env in tqdm(ENVS, desc='environments'):
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
    for arch in tqdm(ARCHITECTURES, desc='geometry'):
        for experiment in EXPERIMENTS:
            experiment_file = f'experiments/{arch}/{experiment}.json'
            defaults = defaults_from_json(experiment_file)
            multiple_seed_experiment(defaults)

def exp_sigma():
    experiment_file = f'experiments/hyperbolic/exp_main.json'
    defaults = defaults_from_json(experiment_file)
    defaults['check_distribution'] = True

    sigmas = [0.5, 1.0, 1.5, 2]
    for sigma in sigmas:
        defaults['gauss_sigma'] = sigma
        defaults['experiment'] = f"{sigma}_sigma_experiment"
        multiple_seed_experiment(defaults)

def epoch_exp():
    experiment_file = f'experiments/hyperbolic/exp_main.json'
    defaults = defaults_from_json(experiment_file)

    epochs = [15, 20, 25, 30, 35, 40, 45, 50]
    for epoch in epochs:
        defaults['epochs'] = epoch
        defaults['experiment'] = f"{epoch}_epoch_experiment"
        multiple_seed_experiment(defaults)

def curvature_exp():
    experiment_file = f'experiments/hyperbolic/exp_main.json'
    defaults = defaults_from_json(experiment_file)

    curvatures = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
    for curvature in curvatures:
        defaults['curvature'] = curvature
        defaults['experiment'] = f"{curvature}_curvature_experiment"
        multiple_seed_experiment(defaults)


if __name__ == '__main__':
    main()

