import distrax
import jax
import jax.numpy as jnp
import optax
import rlax
from flax import nnx

from agents.agent import Agent
from neural_networks import get_network_class

@nnx.jit(static_argnames=['post_process_fn'])
def q_loss_fn(model, states, actions, targets, post_process_fn):
    q_values = post_process_fn(model(states))
    q_selected = jnp.take_along_axis(q_values, actions[:, None], axis=-1).squeeze(-1)
    td_errors = q_selected - targets
    loss = jnp.mean(optax.squared_error(td_errors))
    return loss, {'errors': jnp.abs(td_errors)}

@nnx.jit(static_argnames=['post_process_fn'])
def categorical_loss_fn(model, states, actions, targets, post_process_fn):
    distribution = post_process_fn(model(states))
    selected_distribution = jnp.take_along_axis(distribution, actions[:, None], axis=-1).squeeze(-1)

    actual_distribution = distrax.Categorical(probs=selected_distribution)
    target_distribution = distrax.Categorical(probs=targets)

    error = target_distribution.kl_divergence(actual_distribution)
    loss = jnp.mean(error)
    return loss, {'errors': error}

def direct_model_post(out, atoms):
    q_values = out
    if atoms == 1:
        post_out = q_values
    else:
        q_values = q_values.reshape((q_values.shape[0], atoms, -1))
        post_out = nnx.softmax(q_values, axis=-1)
    return post_out

def duelling_model_post(out, atoms):
    values, advantages = out
    if atoms == 1:
        post_out = values + (advantages - advantages.mean())
    else:
        advantages = advantages.reshape((advantages.shape[0], atoms, -1))
        q_values = values + (advantages - advantages.mean(axis=1, keepdims=True))
        post_out = nnx.softmax(q_values, axis=-1)
    return post_out

class DQNPolicy(nnx.Module):
    def __init__(self, obs_shape, n_actions, rng, config):
        self.model = get_network_class(config)(obs_shape[-1], n_actions, rng, config)
        self.post_processing_fn = nnx.static(duelling_model_post if config.duelling else direct_model_post)
        self.atoms = jnp.arange(config.atoms) # WRONG

        self.noisy_nets = nnx.static(config.noisy_nets)
        self.epsilon = nnx.static(config.epsilon)
        self.distributional = nnx.static(config.distributional)


    def __call__(self, observations, key):
        out = self.model(observations, key) if self.noisy_nets else self.model(observations)
        post_processed_out = self.post_processing_fn(out, self.atoms)
        q_values = self.atoms @ post_processed_out if self.distributional else post_processed_out
        action = jnp.argmax(q_values, axis=-1)

        if self.epsilon > 0.0:
            key1, key2 = jax.random.split(key)
            p = jax.random.uniform(key1, shape=observations.shape[0])
            action = jnp.where(p < self.epsilon, action,
                               jax.random.randint(key2, shape=(q_values.shape[0],),
                                                  minval=0, maxval=q_values.shape[-1]))
        return action, {}

class DQNAgent(Agent):
    def __init__(self, obs_shape, n_actions, rngs, config):
        super().__init__(obs_shape, n_actions, rngs, config)
        self.policy = DQNPolicy(obs_shape, n_actions, rngs, config)
        self.support = jnp.linspace(config.v_min, config.v_max, config.atoms)

        if config.ddqn:
            self.target_model = get_network_class(config)(obs_shape[-1], n_actions, rngs, config)
            nnx.update(self.target_model, nnx.state(self.policy.model))
        else:
            self.target_model = self.policy.model

        self.loss_fn = categorical_loss_fn if config.distributional else q_loss_fn

        self.optimizer = nnx.Optimizer(self.policy.model, optax.adam(learning_rate=self.config.learning_rate), wrt=nnx.Param)

    @staticmethod
    @nnx.jit(static_argnames=['loss_fn', 'post_process_fn'])
    def train_step(model, optimizer, targets, states, actions, loss_fn, post_process_fn):
        grads, aux = nnx.grad(loss_fn, has_aux=True)(model, states, actions, targets)
        optimizer.update(model, grads)  # in-place updates
        return aux['errors']

    def update(self, buffer, rngs):
        # batches_per_epoch = buffer.size // self.config.batch_size
        for _ in range(self.config.n_epochs):
            states, actions, rewards, next_states, discounts, idxs = buffer.sample_batch(self.config.batch_size, rngs())

            next_actions, _ = self.policy(next_states, rngs())

            targets = self.policy.post_processing_fn(self.target_model(next_states, rngs()))
            greedy_targets = jnp.take_along_axis(targets, next_actions[:, None], axis=-1).squeeze(-1)
            if not self.config.distributional:
                targets = rewards + discounts * greedy_targets
            else:
                target_prob_distribution = greedy_targets
                target_support = rewards + discounts * self.support
                target_support = target_support.clamp(min=self.config.v_min, max=self.config.v_max)
                targets = jnp.interp(self.support, target_support, target_prob_distribution)


            errors = self.train_step(self.policy.model, self.optimizer, targets, states, actions,
                                     self.loss_fn, self.policy.post_processing_fn)
            # Priority replay
            buffer.update_priorities(idxs, errors)

            # DDQN
            new_target_params = optax.incremental_update(nnx.state(self.policy.model), nnx.state(self.target_model),
                                                         self.config.polyak_tau)
            nnx.update(self.target_model, new_target_params)