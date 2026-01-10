from pathlib import Path

from flax import nnx
import orbax.checkpoint as ocp

class ModelSerializer:
    def __init__(self, config):
        self.weights_logdir = f'{config.logging_dir}/model'
        Path(self.weights_logdir).mkdir(parents=True, exist_ok=True)
        self.checkpointer = ocp.PyTreeCheckpointer()
        relative_path = f'{self.weights_logdir}/state.ckpt'
        self.ckpt_path = Path(relative_path).resolve()

    def ckpt_exists(self):
        return self.ckpt_path.exists()

    def save(self, model):
        state = nnx.state(model)

        self.checkpointer.save(self.ckpt_path, state)

    def load(self, model):
        state = nnx.state(model)
        restored_state = self.checkpointer.restore(self.ckpt_path, item=state)
        nnx.update(model, restored_state)
        return model

