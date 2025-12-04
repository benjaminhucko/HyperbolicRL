import optax
from flax import nnx

from hypax.manifolds import PoincareBall
from hypax.manifolds.curvature import Curvature
from hypax.opt import riemannian_adam

from networks.euclidean import Critic, ActorCritic
from networks.hybrid import HybActorCritic, HybCritic
from networks.hyperbolic import HCritic, HActorCritic
from networks.hyper import HyperCritic, HyperActorCritic


def make_network(in_shape, out_shape, rngs, config):
    match config.geometry:
        case 'euclidean':
            critic = Critic
            actor_critic = ActorCritic
        case 'hyperbolic':
            critic = HCritic
            actor_critic = HActorCritic
        case 'hybrid':
            if config.hyper or config.hyperpp:
                critic = HyperCritic
                actor_critic = HyperActorCritic
            else:
                critic = HybCritic
                actor_critic = HybActorCritic
        case _:
            raise NotImplementedError(f'unsupported geometry {config.geometry}')

    network_inputs = {'in_channels': in_shape,
                      'n_actions': out_shape,
                      'rngs': rngs,
                      'config': config}

    if config.geometry != 'euclidean':
        init_val = 1.0 if config.learn_curvature else 0.1
        network_inputs['manifold'] = PoincareBall(Curvature(init_val, learnable=config.learn_curvature))
    if not config.duelling and config.strategy == 'dqn':
        return critic(**network_inputs)
    else:
        return actor_critic(**network_inputs)


def make_optim(model, config):
    if config.geometry == 'euclidean':
        optimizer = optax.adam(learning_rate=config.learning_rate)
    else:
        optimizer = riemannian_adam(config.learning_rate)
    return nnx.Optimizer(model, optimizer, wrt=nnx.Param)