from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from pyexpat import features


def euclidean_activation_fn_factory(activation_name):
    if activation_name == 'relu':
        return nnx.relu
    elif activation_name == 'elu':
        return nnx.elu
    elif activation_name == 'gelu':
        return nnx.gelu
    else:
        raise ValueError(f'Unknown activation function {activation_name}')

class ImpalaResidualBlock(nnx.Module):
    def __init__(self, num_filters, cnn_args, config):
        super().__init__()
        self.conv1 = nnx.Conv(num_filters, num_filters, **cnn_args)
        self.conv2 = nnx.Conv(num_filters, num_filters, **cnn_args)
        self.activation_fn = euclidean_activation_fn_factory(config.activation)

    def __call__(self, x):
        out = self.activation_fn(self.conv1(x))
        out = self.activation_fn(self.conv2(out))
        return x + out

class ImpalaFeatureExtractor(nnx.Module):
    def __init__(self, in_channels, hidden_channels, rngs, config):
        super().__init__()
        cnn_args = {'kernel_size': config.kernel_size, 'strides': config.stride,
                    'rngs': rngs}

        self.activation_fn = euclidean_activation_fn_factory(config.activation)
        # self.pool = HMaxPool2D()
        self.conv = nnx.Conv(in_channels, hidden_channels, **cnn_args)
        self.impala_layers = nnx.List([ImpalaResidualBlock(hidden_channels, cnn_args, config)
                                       for _ in range(config.n_conv)])

    def __call__(self, x):
        x = self.conv(x)
        # x = self.pool(x)
        for layer in self.impala_layers:
            x = layer(x)
        x = self.activation_fn(x)
        return x

class CNN(nnx.Module):
    def __init__(self, in_channels, out_channels, rngs, config):
        super().__init__()
        cnn_args = {'kernel_size': config.kernel_size, 'strides': config.stride, 'rngs': rngs}
        self.activation_fn = euclidean_activation_fn_factory(config.activation)
        layers = []
        if config.n_conv == 1:
            layers.append(nnx.Conv(in_channels, out_channels, **cnn_args))
        else:
            layers.append(nnx.Conv(in_channels, config.hidden_channels, **cnn_args))
            hidden_layers = [nnx.Conv(config.hidden_channels, config.hidden_channels, **cnn_args)
                             for _ in range(config.n_conv - 2)]
            layers.extend(hidden_layers)
            layers.append(nnx.Conv(config.hidden_channels, out_channels, **cnn_args))
        self.layers = nnx.List(layers)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        return x

class MLP(nnx.Module):
    def __init__(self, in_channels, out_channels, rngs, config):
        self.activation_fn = euclidean_activation_fn_factory(config.activation)
        self.layers = nnx.List()

        if config.n_linear == 1:
            self.layers.append(nnx.Linear(in_channels, out_channels, rngs=rngs))
        else:
            self.layers.append(nnx.Linear(in_channels, config.hidden_channels, rngs=rngs))
            hidden_layers = [nnx.Linear(config.hidden_channels, config.hidden_channels, rngs=rngs)
                             for _ in range(config.n_linear - 2)]
            self.layers.extend(hidden_layers)
            self.layers.append(nnx.Linear(config.hidden_channels, out_channels, rngs=rngs))

    def __call__(self, x, key=None, analyze=False):
        for layer in self.layers[:-1]:
            features = layer(x)
            x = self.activation_fn(features)
        out = self.layers[-1](x)

        if analyze:
            return out, features
        return out

class NoisyMLP(nnx.Module):
    def __init__(self, in_channels, out_channels, rngs, config):
        self.activation_fn = euclidean_activation_fn_factory(config.activation)

        self.layers = nnx.List()
        mlp_args = {'rngs': rngs, 'config': config}

        if config.n_linear == 1:
            self.layers.append(NoisyLinear(in_channels, out_channels, **mlp_args))
        else:
            self.layers.append(NoisyLinear(in_channels, config.hidden_channels, **mlp_args))
            hidden_layers = [NoisyLinear(config.hidden_channels, config.hidden_channels, **mlp_args)
                             for _ in range(config.n_linear - 2)]
            self.layers.extend(hidden_layers)
            self.layers.append(NoisyLinear(config.hidden_channels, out_channels, **mlp_args))

    def __call__(self, x, layer_key, analyze=False):
        keys = jax.random.split(layer_key, len(self.layers))
        for layer, key in zip(self.layers[:-1], keys[:-1]):
            features = layer(x, key)
            x = self.activation_fn(features)

        out = self.layers[-1](x, keys[-1])

        if analyze:
            return out, features

        return out


class NoisyLinear(nnx.Module):
    def __init__(self, in_features, out_features, rngs, config):
        self.in_features, self.out_features = in_features, out_features
        key_mu = rngs.params()
        self.w_mu = nnx.Param(jax.random.uniform(key_mu, (in_features, out_features),
            minval=-1.0 / jnp.sqrt(in_features),
            maxval=1.0 / jnp.sqrt(in_features)
        ))
        self.w_sigma = nnx.Param(jnp.ones((in_features, out_features)) * (config.std_init / jnp.sqrt(in_features)))
        self.b_mu = nnx.Param(jnp.zeros((out_features,)))
        self.b_sigma = nnx.Param(jnp.ones((out_features,)) * (config.std_init / jnp.sqrt(out_features)))

    def f(self, x):
        return jnp.sign(x) * jnp.sqrt(jnp.abs(x))

    def sample_noise(self, key):
        key_in, key_out = jax.random.split(key)
        eps_in = self.f(jax.random.normal(key_in, (self.in_features,)))
        eps_out = self.f(jax.random.normal(key_out, (self.out_features,)))
        noise_w = jnp.outer(eps_in, eps_out)
        noise_b = eps_out
        return noise_w, noise_b

    def __call__(self, x, key):
        noise_w, noise_b = self.sample_noise(key)
        w = self.w_mu + self.w_sigma * noise_w
        b = self.b_mu + self.b_sigma * noise_b
        return x @ w + b


class ActorCritic(nnx.Module):
    def __init__(self, in_channels, n_actions, rngs, config):
        self.atoms = config.atoms
        self.feature_extractor = ImpalaFeatureExtractor(in_channels, config.hidden_channels, rngs, config)
        actor_atoms = self.atoms if config.categorical_actor else 1
        critic_atoms = self.atoms
        mlp = MLP if not config.noisy_nets else NoisyMLP
        self.actor = mlp(config.hidden_channels * 100, n_actions * actor_atoms, rngs, config)
        self.critic = mlp(config.hidden_channels * 100, critic_atoms, rngs, config)

    def __call__(self, x, key=None, analyze=False):
        features = self.feature_extractor(x)
        features = features.reshape(features.shape[0], -1)
        actor = self.actor(features, key, analyze)
        critic = self.critic(features, key, analyze)

        if analyze:
            return actor[0], critic[0], {'visual': features, 'actor': actor[1], 'critic': critic[1]}

        return actor, critic

class Critic(nnx.Module):
    def __init__(self, in_channels, n_actions, rngs, config):
        self.atoms = config.atoms
        self.feature_extractor = CNN(in_channels, config.hidden_channels, rngs, config)
        mlp = MLP if not config.noisy_nets else NoisyMLP
        self.mlp = mlp(config.hidden_channels * 100, n_actions * self.atoms, rngs, config)

    def __call__(self, x, key=None, analyze=False):
        x = self.feature_extractor(x)
        features = x.reshape((x.shape[0], -1))
        x = self.mlp(features, key, analyze)
        if analyze:
            return x[0], {'visual': features, 'critic': x[1]}
        return x