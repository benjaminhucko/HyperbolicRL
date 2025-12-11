from collections import defaultdict
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import rlax
from flax import nnx
from jax import grad, vmap
from matplotlib import pyplot as plt

from optimization.norm import normalize


class Analyzer:
    def __init__(self, config):
        self.data = defaultdict(list)
        self.rollout_tracker = 1
        self.track_every = 50 # in rollouts
        self.heatmap_dir = f'analysis/{config.env}/heatmaps/{config.geometry}'
        self.timeseries_dir = f'analysis/{config.env}/plots/{config.geometry}'

        self.timeseries_data = {'grad': defaultdict(list), 'srank': defaultdict(list)}
        self.timeseries_legend = {}

        Path(self.heatmap_dir).mkdir(parents=True, exist_ok=True)
        Path(self.timeseries_dir).mkdir(parents=True, exist_ok=True)

    def make_heatmap(self, data):
        uid = f'{self.heatmap_dir}/{self.rollout_tracker}'
        np.save(uid, data)
        fig = plt.figure()
        plt.imshow(data, cmap='viridis')
        plt.title('Heatmap of covariance in gradients')
        plt.savefig(f'{uid}.png')
        plt.close(fig)

    def make_timeseries(self, key, data):
        uid = f'{self.timeseries_dir}/{key}'
        x_ticks = np.linspace(0, self.rollout_tracker, len(data))
        np.save(uid, data)
        total_rank = self.timeseries_legend[key]
        fig = plt.figure()
        plt.plot(x_ticks, data, linestyle='-')
        plt.xlabel('rollout')
        plt.ylabel('effective rank')
        plt.title(f'plot of effective rank over time (total rank {total_rank})')
        plt.savefig(f'{uid}.png')
        plt.close(fig)

    def add_to_dict(self, data, path, value):
        if path not in data:
            data[path] = [value]
        else:
            data[path].append(value)

    def track_grads(self, grads):
        # GRADS: [Batch, dim]
        grads = jnp.concatenate(grads, axis=0)
        norms = jnp.linalg.norm(grads, axis=-1, keepdims=True)
        cov = (grads @ grads.T) / (norms @ norms.T)
        self.make_heatmap(cov)


    def track_effective_rank(self, features, sigma=0.01):
        for key, value in features.items():
            s_ranks = []
            for batch_values in value:
                s_vals = jnp.linalg.svd(batch_values, compute_uv=False)
                s_vals = jnp.sort(s_vals, axis=0, descending=True)
                s_vals = s_vals / jnp.sum(s_vals)
                cumulative_s_vals = jnp.cumsum(s_vals)
                b_s_rank = jnp.argmax(cumulative_s_vals >= 1 - sigma)
                s_ranks.append(b_s_rank)

                if key not in self.timeseries_legend:
                    self.timeseries_legend[key] = len(s_vals)

            s_rank = sum(s_ranks) / len(s_ranks)
            self.add_to_dict(self.timeseries_data['srank'], key, s_rank)


    def step(self, stats):
        if 'grads' in stats:
            self.track_grads(stats.pop('grads'))
        self.track_effective_rank(stats.pop('embeddings'))
        self.rollout_tracker += 1


    def clear_all(self):
        self.data = {}
        self.timeseries_data = {{'grads': {}, 'srank': {}}}

    def plot_all(self):
        for name, data in self.timeseries_data.items():
            for key_path, values in data.items():
                self.make_timeseries(key_path, values)

    @staticmethod
    def process_raw_grads(grads):
        processed_grads = []
        flattened, _ = jax.tree_util.tree_flatten(grads)
        for value in flattened:
            value = value.reshape(value.shape[0], -1)
            processed_grads.append(value)
        processed_grads = jnp.concatenate(processed_grads, axis=1)
        return processed_grads

    @staticmethod
    def append(stats, aux):
        for k, v in aux.items():
            if k in {"embeddings"}:
                if 'embeddings' not in stats:
                    stats['embeddings'] = {}
                for k2, v2 in v.items():
                    stats['embeddings'].setdefault(k2, []).append(v2)
            else:
                stats[k].append(v)
        return stats

    def analyze_grads(self):
        return self.rollout_tracker % self.track_every == 0

    @staticmethod
    @nnx.jit(static_argnames=['value_loss_fn', 'clip_threshold', 'value_weight', 'regularization'])
    def ppo_loss_analysis(model, returns, advantages, observations, actions, old_log_probs,
                       clip_threshold, regularization, value_weight, value_loss_fn):

        batch_size = returns.shape[0]
        returns, advantages, observations, actions, old_log_probs = (jnp.expand_dims(val, axis=1) for val in
                                             [returns, advantages, observations, actions, old_log_probs])
        def batch_loss_fn(model, returns, advantages, observations, actions, old_log_probs):
            out = model(observations, None, analyze=False)
            action_logits, values = out[0], out[1]

            policy = distrax.Categorical(action_logits)
            log_probs = policy.log_prob(actions)
            log_ratio = log_probs - old_log_probs
            ratio = jnp.exp(log_ratio)
            normalized_advantages = normalize(advantages)

            clipped_ratios_t = jnp.clip(ratio, 1. - clip_threshold, 1. + clip_threshold)
            policy_loss = -jnp.fmin(ratio * normalized_advantages, clipped_ratios_t * normalized_advantages)

            entropy_loss = policy.entropy()

            value_loss = value_loss_fn(values, returns)
            loss = policy_loss - regularization * entropy_loss + value_weight * value_loss
            loss = loss / batch_size
            return loss.squeeze()
        loss_fn = nnx.vmap(nnx.grad(batch_loss_fn), in_axes=(None, 0, 0, 0, 0, 0), out_axes=0)
        return loss_fn(model, returns, advantages, observations, actions, old_log_probs)

