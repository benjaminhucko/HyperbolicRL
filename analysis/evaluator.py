from pathlib import Path

import jax.numpy as jnp

class Evaluator:
    def __init__(self, config, sticky_action=False):
        eval_mode = 'sticky_action' if sticky_action else 'normal'
        self.eval_logdir = f'{config.logging_dir}/eval/{eval_mode}/'
        Path(self.eval_logdir).mkdir(parents=True, exist_ok=True)


    def log(self, returns):
        jnp.save(self.eval_logdir, returns)


