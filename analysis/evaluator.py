from pathlib import Path

import jax.numpy as jnp

class Evaluator:
    def __init__(self, config):
        self.eval_logdir = f'{config.logging_dir}/eval/'
        Path(self.eval_logdir).mkdir(parents=True, exist_ok=True)

    def log(self, sticky_sigma, returns):
        jnp.savez(f'{self.eval_logdir}/{sticky_sigma}', returns)


