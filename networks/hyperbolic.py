from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from pyexpat import features

import hypax.nn as hnn
from hypax.array import ManifoldArray
from hypax.nn import HConvolution2D, HLinear


def hyperbolic_activation_fn_factory(activation_name):
    if activation_name == 'relu':
        return hnn.hrelu
    elif activation_name == 'elu':
        return hnn.helu
    else:
        raise ValueError(f'Unknown activation function {activation_name}')

class HCNN(nnx.Module):
    def __init__(self, in_channels, out_channels, manifold, rngs, config):
        super().__init__()
        cnn_args = {'kernel_size': config.kernel_size, 'stride': config.stride, 'manifold': manifold, 'rngs': rngs,
                    'padding': 1}
        self.activation_fn = hyperbolic_activation_fn_factory(config.activation)
        layers = []
        if config.n_conv == 1:
            layers.append(hnn.HConvolution2D(in_channels, out_channels, **cnn_args))
        else:
            layers.append(hnn.HConvolution2D(in_channels, config.hidden_channels, **cnn_args))
            hidden_layers = [hnn.HConvolution2D(config.hidden_channels, config.hidden_channels, **cnn_args)
                             for _ in range(config.n_conv - 2)]
            layers.extend(hidden_layers)
            layers.append(hnn.HConvolution2D(config.hidden_channels, out_channels, **cnn_args))
        self.layers = nnx.List(layers)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x, axis=1)
        x = self.layers[-1](x)
        return x

class HMLP(nnx.Module):
    def __init__(self, in_channels, out_channels, manifold, rngs, config):
        self.activation_fn = hyperbolic_activation_fn_factory(config.activation)

        mlp_args = {'manifold': manifold, 'rngs': rngs}
        self.layers = nnx.List()
        self.analyze = config.analyze

        if config.n_linear == 1:
            self.layers.append(HLinear(in_channels, out_channels, **mlp_args))
        else:
            self.layers.append(HLinear(in_channels, config.hidden_channels, **mlp_args))
            hidden_layers = [HLinear(config.hidden_channels, config.hidden_channels, **mlp_args)
                             for _ in range(config.n_linear - 2)]
            self.layers.extend(hidden_layers)
            self.layers.append(HLinear(config.hidden_channels, out_channels, **mlp_args))

    def __call__(self, x, key=None):
        for layer in self.layers[:-1]:
            features = layer(x)
            x = self.activation_fn(features)
        out = self.layers[-1](x)
        if self.analyze:
            return out, features

        return out

class HNoisyMLP(nnx.Module):
    def __init__(self, in_channels, out_channels, manifold, rngs, config):
        self.activation_fn = hyperbolic_activation_fn_factory(config.activation)
        mlp_args = {'manifold': manifold, 'rngs': rngs, 'config': config}
        self.analyze = config.analyze

        self.layers = nnx.List()
        if config.n_linear == 1:
            self.layers.append(HNoisyLinear(in_channels, out_channels, **mlp_args))
        else:
            self.layers.append(HNoisyLinear(in_channels, config.hidden_channels, **mlp_args))
            hidden_layers = [HNoisyLinear(config.hidden_channels, config.hidden_channels, **mlp_args)
                             for _ in range(config.n_linear - 2)]
            self.layers.extend(hidden_layers)
            self.layers.append(HNoisyLinear(config.hidden_channels, out_channels, **mlp_args))

    def __call__(self, x, layer_key):
        keys = jax.random.split(layer_key, len(self.layers))
        for layer, key in zip(self.layers[:-1], keys[:-1]):
            features = layer(x, key)
            x = self.activation_fn(features)
        out = self.layers[-1](x, keys[-1])

        if self.analyze:
            return out, features
        return out


class HImpalaResidualBlock(nnx.Module):
    def __init__(self, num_filters, cnn_args, config):
        super().__init__()

        self.conv1 = HConvolution2D(num_filters, num_filters, **cnn_args)
        self.conv2 = HConvolution2D(num_filters, num_filters, **cnn_args)
        self.activation_fn = hyperbolic_activation_fn_factory(config.activation)

    def __call__(self, x):
        out = self.activation_fn(self.conv1(x), axis=1)
        out = self.activation_fn(self.conv2(out), axis=1)
        out = x.manifold.mobius_add(out.data, x.data, axis=1)
        out = x.replace(data=out)
        return out

class HImpalaFeatureExtractor(nnx.Module):
    def __init__(self, in_channels, hidden_channels, manifold, rngs, config):
        super().__init__()
        cnn_args = {'kernel_size': config.kernel_size, 'stride': config.stride,
                    'manifold': manifold, 'padding': 1, 'rngs': rngs}

        self.activation_fn = hyperbolic_activation_fn_factory(config.activation)
        # self.pool = HMaxPool2D()
        self.conv = HConvolution2D(in_channels, hidden_channels, **cnn_args)
        self.impala_layers = nnx.List([HImpalaResidualBlock(hidden_channels, cnn_args, config) for _ in range(2)])

    def __call__(self, x):
        x = self.conv(x)
        # x = self.pool(x)
        x = self.activation_fn(x, axis=1)
        for layer in self.impala_layers:
            x = layer(x)
        x = self.activation_fn(x, axis=1)
        return x

class HNoisyLinear(nnx.Module):
    def __init__(self, in_features, out_features, rngs, config):
        raise NotImplemented

class HActorCritic(nnx.Module):
    def __init__(self, in_channels, n_actions, manifold, rngs, config):
        self.atoms = config.atoms
        self.manifold = manifold
        self.analyze = config.analyze

        self.feature_extractor = HImpalaFeatureExtractor(in_channels, config.hidden_channels, self.manifold, rngs, config)
        # self.feature_extractor = HCNN(in_channels, config.hidden_channels, manifold, rngs, config)
        actor_atoms = self.atoms if config.categorical_actor else 1
        critic_atoms = self.atoms

        mlp = HMLP if not config.noisy_nets else HNoisyMLP
        self.actor = mlp(config.hidden_channels * 100, n_actions * actor_atoms, self.manifold, rngs, config)
        self.critic = mlp(config.hidden_channels * 100, critic_atoms, self.manifold, rngs, config)

    def __call__(self, x, key=None):
        x = self.manifold.expmap(x, axis=1)
        x = ManifoldArray(x, self.manifold)
        x = self.feature_extractor(x)
        features = x.flatten(manifold_axis=1)
        actor = self.actor(features, key)
        critic = self.critic(features, key)
        if self.analyze:
            return actor[0].data, critic[0].data, {'visual': features.data, 'actor': actor[1].data,
                                                   'critic': critic[1].data}

        return actor.data, critic.data

class HCritic(nnx.Module):
    def __init__(self, in_channels, n_actions, manifold, rngs, config):
        self.atoms = config.atoms
        self.manifold = manifold
        self.analyze = config.analyze

        self.feature_extractor = HImpalaFeatureExtractor(in_channels, config.hidden_channels, manifold, rngs, config)
        mlp = HMLP if not config.noisy_nets else HNoisyMLP
        self.mlp = mlp(config.hidden_channels * 100, n_actions * self.atoms, manifold, rngs, config)

    def __call__(self, x, key=None):
        x = self.manifold.expmap(x, axis=1)
        x = ManifoldArray(data=x, manifold=self.manifold)
        x = self.feature_extractor(x)
        features = x.flatten(manifold_axis=1)
        x = self.mlp(features, key)
        if self.analyze:
            return x[0].data, {'visual': features.data, 'critic': x[1].data}

        return x.data