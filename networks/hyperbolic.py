from functools import partial

import jax
import jax.numpy as jnp
import hypax.nn as hnn
from flax import nnx
from hypax.array import ManifoldArray


def _activation_fn_factory(activation_name):
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
        self.activation_fn = _activation_fn_factory(config.activation)
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
        self.activation_fn = _activation_fn_factory(config.activation)
        linear_layer_cls = get_linear_class(config)
        self.noisy = config.noisy_nets

        layers = []
        mlp_args = {'manifold': manifold, 'rngs': rngs}
        if config.n_linear == 1:
            layers.append(linear_layer_cls(in_channels, out_channels, **mlp_args))
        else:
            layers.append(linear_layer_cls(in_channels, config.hidden_channels, **mlp_args))
            hidden_layers = [linear_layer_cls(config.hidden_channels, config.hidden_channels, **mlp_args)
                             for _ in range(config.n_linear - 2)]
            layers.extend(hidden_layers)
            layers.append(linear_layer_cls(config.hidden_channels, out_channels, **mlp_args))
        self.layers = nnx.List(layers)

        self.forward = nnx.static(self.noisy_forward if config.noisy_nets else self.normal_forward)

    def noisy_forward(self, x, layer_key):
        keys = jax.random.split(layer_key, len(self.layers))
        for layer, key in zip(self.layers[:-1], keys[:-1]):
            x = layer(x, key)
            x = self.activation_fn(x)
        x = self.layers[-1](x, keys[-1])
        return x

    def normal_forward(self, x, *args):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.layers[-1](x)
        return x

    def __call__(self, x, key=None):
        return self.forward(x, key)


class HActorCritic(nnx.Module):
    def __init__(self, in_channels, n_actions, manifold, rngs, config):
        self.atoms = config.atoms
        self.manifold = manifold
        self.feature_extractor = HCNN(in_channels, config.hidden_channels, self.manifold, rngs, config)

        self.actor = HMLP(config.hidden_channels * 100, n_actions * self.atoms, self.manifold, rngs, config)
        self.critic = HMLP(config.hidden_channels * 100, self.atoms, self.manifold, rngs, config)

    def __call__(self, x, key=None):
        x = self.manifold.expmap(x, axis=1)
        x = ManifoldArray(x, self.manifold)
        x = self.feature_extractor(x)
        x = x.flatten(manifold_axis=1)
        actor = self.actor(x, key)
        critic = self.critic(x, key)
        return actor.data, critic.data

class HCritic(nnx.Module):
    def __init__(self, in_channels, n_actions, manifold, rngs, config):
        self.atoms = config.atoms
        self.manifold = manifold
        self.feature_extractor = HCNN(in_channels, config.hidden_channels, manifold, rngs, config)
        self.mlp = HMLP(config.hidden_channels * 100, n_actions * self.atoms, manifold, rngs, config)

    def __call__(self, x, key=None):
        x = self.manifold.expmap(x, axis=1)
        x = ManifoldArray(data=x, manifold=self.manifold)
        x = self.feature_extractor(x)
        x = x.flatten(manifold_axis=1)
        x = self.mlp(x, key)
        return x.data

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

def get_linear_class(config):
    if config.noisy_nets:
        return partial(NoisyLinear, config=config)
    else:
        return hnn.HLinear
