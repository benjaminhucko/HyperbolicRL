from flax import nnx

from hypax.array import ManifoldArray
from networks.euclidean import CNN
from networks.hyperbolic import HMLP



class HybActorCritic(nnx.Module):
    def __init__(self, in_channels, n_actions, manifold, rngs, config):
        self.atoms = config.atoms
        self.manifold = manifold
        self.feature_extractor = CNN(in_channels, config.hidden_channels, rngs, config)

        self.actor = HMLP(config.hidden_channels * 100, n_actions * self.atoms, self.manifold, rngs, config)
        self.critic = HMLP(config.hidden_channels * 100, self.atoms, self.manifold, rngs, config)

    def __call__(self, x, key=None):
        x = self.feature_extractor(x)

        x = x.reshape(x.shape[0], -1)
        x = self.manifold.expmap(x)
        x = ManifoldArray(x, self.manifold)

        actor = self.actor(x, key)
        critic = self.critic(x, key)
        return actor.data, critic.data

class HybCritic(nnx.Module):
    def __init__(self, in_channels, n_actions, manifold, rngs, config):
        self.atoms = config.atoms
        self.manifold = manifold
        self.feature_extractor = CNN(in_channels, config.hidden_channels, rngs, config)
        self.mlp = HMLP(config.hidden_channels * 100, n_actions * self.atoms, manifold, rngs, config)

    def __call__(self, x, key=None):
        x = self.feature_extractor(x)

        x = x.reshape(x.shape[0], -1)
        x = self.manifold.expmap(x)
        x = ManifoldArray(x, self.manifold)

        x = self.mlp(x, key)
        return x.data

