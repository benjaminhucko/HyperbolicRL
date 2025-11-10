import jax
import jax.numpy as jnp
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
        self.critic = MLP(config.hidden_channels * 100, 1, rngs, config)

    def __call__(self, x):
        features = self.feature_extractor(x)
        features = features.reshape(features.shape[0], -1)
        features = self.activation_fn(features)
        actor = self.actor(features)
        critic = self.critic(features)

        return actor, critic

class Critic(nnx.Module):
    output_features: int
    hidden_features: int = 64
    kernel_size: int = 3
    stride: int = 1
    num_layers: int = 3

    def __init__(self, in_channels, n_actions, rngs, config):
        self.feature_extractor = CNN(in_channels, config.hidden_channels, rngs, config)
        self.linear1 = nnx.Linear(config.hidden_channels * 100, 512, rngs=rngs)
        self.linear2 = nnx.Linear(512, n_actions, rngs=rngs)

    def __call__(self, x):
        x = self.feature_extractor(x)
        x = nnx.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        return x

class NoisyLinear(nnx.Module):
    def __init__(self, in_features, out_features, rngs, config):
        self.in_features, self.out_features = in_features, out_features
        # Mean and std parameters
        self.mu_w = jax.random.uniform(rngs(), (out_features, in_features),
            minval=-1.0 / jnp.sqrt(in_features),
            maxval=1.0 / jnp.sqrt(in_features)
        )
        self.sigma_w = jnp.ones((out_features, in_features)) * (config.std_init / jnp.sqrt(in_features))
        self.mu_b = jnp.zeros((out_features,))
        self.sigma_b = jnp.ones((out_features,)) * (config.std_init / jnp.sqrt(out_features))

    def f(self, x):
        """Helper for factorized noise (Fortunato et al., 2017)"""
        return jnp.sign(x) * jnp.sqrt(jnp.abs(x))

    def sample_noise(self, key):
        """Generate factorized Gaussian noise"""
        key_in, key_out = jax.random.split(key)
        eps_in = self.f(jax.random.normal(key_in, (self.in_features,)))
        eps_out = self.f(jax.random.normal(key_out, (self.out_features,)))
        noise_w = jnp.outer(eps_out, eps_in)
        noise_b = eps_out
        return noise_w, noise_b

    def __call__(self, x, key):
        noise_w, noise_b = self.sample_noise(key)
        w = self.mu_w + self.sigma_w * noise_w
        b = self.mu_b + self.sigma_b * noise_b
        return x @ w.T + b


def get_network_class(config):
    if config.duelling:
        return ActorCritic
    else:
        return Critic