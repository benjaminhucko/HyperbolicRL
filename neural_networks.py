import jax
from flax import nnx

def activation_fn_factory(activation_name):
    if activation_name == 'relu':
        return nnx.relu
    elif activation_name == 'elu':
        return nnx.elu
    elif activation_name == 'gelu':
        return nnx.gelu
    else:
        raise ValueError(f'Unknown activation function {activation_name}')

class CNN(nnx.Module):
    def __init__(self, in_channels, out_channels, rngs, config):
        super().__init__()
        cnn_args = {'kernel_size': config.kernel_size, 'strides': config.stride, 'rngs': rngs}
        self.activation_fn = activation_fn_factory(config.activation)
        self.input_layer = nnx.Conv(in_channels, config.hidden_channels, **cnn_args)
        self.hidden_layers = nnx.List([nnx.Conv(config.hidden_channels, config.hidden_channels, **cnn_args)])
        self.output_layer = nnx.Conv(config.hidden_channels, out_channels, **cnn_args)

    def __call__(self, x):
        x = self.input_layer(x)
        x = self.activation_fn(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation_fn(x)

        x = self.output_layer(x)
        return x

class MLP(nnx.Module):
    def __init__(self, in_channels, out_channels, rngs, config):
        self.activation_fn = activation_fn_factory(config.activation)
        self.input_layer = nnx.Linear(in_channels, config.hidden_channels, rngs=rngs)
        self.output_layer = nnx.Linear(config.hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        x = self.input_layer(x)
        x = self.activation_fn(x)
        x = self.output_layer(x)
        return x

class ActorCritic(nnx.Module):
    def __init__(self, in_channels, n_actions, rngs, config):
        super().__init__()
        self.feature_extractor = CNN(in_channels, config.hidden_channels, rngs, config)
        self.activation_fn = activation_fn_factory(config.activation)
        self.actor = MLP(config.hidden_channels * 100, n_actions, rngs, config)
        self.critic = MLP(config.hidden_channels * 100, n_actions, rngs, config)

    def __call__(self, x):
        features = self.feature_extractor(x)
        features = features.reshape(features.shape[0], -1)
        features = self.activation_fn(features)
        actor = self.actor(features)
        critic = self.critic(features)

        return actor, critic