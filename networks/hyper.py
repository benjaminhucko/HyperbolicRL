from functools import partial

from flax import nnx
from flax.nnx import RMSNorm

from hypax.array import ManifoldArray
from networks.euclidean import CNN, euclidean_activation_fn_factory, ImpalaResidualBlock, MLP
from networks.hyperbolic import HMLP

import jax.numpy as jnp


class HyperEuclideanPart(nnx.Module):
    def __init__(self, in_channels, out_features, rngs, config):
        super().__init__()
        cnn_args = {'kernel_size': config.kernel_size, 'strides': config.stride, 'rngs': rngs}

        self.activation_fn = euclidean_activation_fn_factory(config.activation)
        self.pool = nnx.max_pool
        self.pool_args = {'window_shape':(3, 3), 'strides':(2, 2)}
        self.conv = nnx.Conv(in_channels, 16, **cnn_args)
        self.impala_layers = nnx.List([ImpalaResidualBlock(16, cnn_args) for _ in range(2)])
        self.linear = nnx.Linear(1600, out_features, rngs=rngs)
        self.apply_hyperpp_improvements = config.hyperpp


        if config.hyperpp:
            self.rms_norm = RMSNorm(num_features=out_features, rngs=nnx.Rngs(0))
            self.alpha = nnx.Param(0.95)

    def __call__(self, x):
        x = self.conv(x)
        # x = self.pool(x, **self.pool_args)
        for layer in self.impala_layers:
            x = layer(x)
        x = self.activation_fn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.activation_fn(x)
        x = self.linear(x)

        if self.apply_hyperpp_improvements:
            x = self.rms_norm(x)
            x = nnx.tanh(x)
            x *= self.alpha
        else:
            x /= jnp.sqrt(x.shape[-1])

        return x

class HyperActorCritic(nnx.Module):
    def __init__(self, in_channels, n_actions, manifold, rngs, config):
        self.atoms = config.atoms
        self.feature_extractor = HyperEuclideanPart(in_channels, config.hidden_features, rngs, config)

        if config.baseline:
            self.actor = MLP(config.hidden_features, n_actions * self.atoms, rngs, config)
            self.critic = MLP(config.hidden_features, self.atoms, rngs, config)
        else:
            self.manifold = manifold
            self.actor = HMLP(config.hidden_features, n_actions * self.atoms, self.manifold, rngs, config)
            self.critic = HMLP(config.hidden_features, self.atoms, self.manifold, rngs, config)
        self.project = not config.baseline

    def __call__(self, x, key=None):
        x = self.feature_extractor(x)

        if self.project:
            x = self.manifold.expmap(x)
            x = ManifoldArray(x, self.manifold)

        actor = self.actor(x, key)
        critic = self.critic(x, key)
        if self.project:
            actor = actor.data
            critic = critic.data

        return actor, critic

class HyperCritic(nnx.Module):
    def __init__(self, in_channels, n_actions, manifold, rngs, config):
        self.atoms = config.atoms
        self.baseline = config.baseline
        self.feature_extractor = HyperEuclideanPart(in_channels, config.hidden_features, rngs, config)

        if config.baseline:
            self.mlp = MLP(config.hidden_features, n_actions * self.atoms, rngs, config)
        else:
            self.manifold = manifold
            self.mlp = HMLP(config.hidden_features, n_actions * self.atoms, manifold, rngs, config)

        self.project = not config.baseline


    def __call__(self, x, key=None):
        x = self.feature_extractor(x)
        if self.project:
            x = self.manifold.expmap(x)
            x = ManifoldArray(x, self.manifold)

        x = self.mlp(x, key)

        if self.project:
            x = x.data
        return x


def apply_spectral_norm(params):
    pass

# Hyper
# TODO: apply spectral norm: Ask when to apply
#                           apply scaling by sqrt(n) (or gaussian with E[sqrt(n)])

# Hyperpp
# TODO: Categorical value function. (categorical HL-Gauss loss)
#       Hyperboloid model

# MEETING TODO:
# initialize curvature to 1.0
# 1. try differnet environment
# 2. optimization
# 3. UMAP/Trimap analysis
