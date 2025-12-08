from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt


class Analyzer:
    def __init__(self, config):
        self.data = defaultdict(list)
        self.rollout_tracker = 1
        self.track_every = 50 # in rollouts
        self.heatmap_dir = 'analysis/heatmaps'
        self.timeseries_dir = 'analysis/plots'

        self.timeseries_data = {'grad': defaultdict(list), 'srank': defaultdict(list)}
        self.timeseries_legend = {}

        Path(self.heatmap_dir).mkdir(parents=True, exist_ok=True)
        Path(self.timeseries_dir).mkdir(parents=True, exist_ok=True)

    def make_heatmap(self, data):
        fig = plt.figure()
        plt.imshow(data, cmap='viridis')
        plt.title('Heatmap of covariance in gradients')
        plt.savefig(f'{self.heatmap_dir}/{self.rollout_tracker}.png')
        plt.close(fig)

    def make_timeseries(self, key, data):
        x_ticks = np.linspace(0, self.rollout_tracker, len(data))
        data = np.array(data)
        total_rank = self.timeseries_legend[key]
        fig = plt.figure()
        plt.plot(x_ticks, data, linestyle='-')
        plt.xlabel('rollout')
        plt.ylabel('effective rank')
        plt.title(f'plot of effective rank over time (total rank {total_rank})')
        plt.savefig(f'{self.timeseries_dir}/{key}.png')
        plt.close(fig)

    def add_to_dict(self, data, path, value):
        if path not in data:
            data[path] = [value]
        else:
            data[path].append(value)

    def track_grads(self, grads):
        # GRADS: [Batch, dim]
        grads = jnp.stack(grads)
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
            value = value.reshape(-1)
            processed_grads.append(value)
        processed_grads = jnp.concatenate(processed_grads, axis=None)
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
