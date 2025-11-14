from flax import nnx
from hypax.manifolds import PoincareBall
from hypax.opt import riemannian_adam

from networks.euclidean import Critic, ActorCritic
from networks.hyperbolic import HCritic, HActorCritic, CombatibilityLayer


def make_network(in_shape, out_shape, rngs, config):
    if config.geometry == 'euclidean':
        if not config.duelling and config.strategy == 'dqn':
            return Critic(in_shape, out_shape, rngs, config)
        else:
            return ActorCritic(in_shape, out_shape, rngs, config)

    elif config.geometry == 'hyperbolic':
        manifold = PoincareBall(c=1.0)
        if not config.duelling and config.strategy == 'dqn':
            return CombatibilityLayer(HCritic(in_shape, out_shape, manifold, rngs, config))
        else:
            return CombatibilityLayer(HActorCritic(in_shape, out_shape, manifold, rngs, config))



def make_optim(model, config):
    if config.geometry == 'euclidean':
        optimizer = nnx.adam(learning_rate=config.learning_rate)
    elif config.geometry == 'hyperbolic':
         optimizer = riemannian_adam(config.learning_rate)
    else:
        raise ValueError('Unknown geometry {}'.format(config.geometry))
    return nnx.Optimizer(model, optimizer, wrt=nnx.Param)