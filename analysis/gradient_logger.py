from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt


class Analyzer:
    def __init__(self):
        self.data = {}
        self.step_tracker = 0
        self.track_every = 200
        self.n = 0
        self.log_dir = 'analysis/logs'
        self.timeseries_data = {'grad': {}, 'srank': {}}

        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def make_hist(self, path, data):
        plt.hist(data, bins=10, color='skyblue', edgecolor='black')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of grad updates')
        plt.savefig(f'{self.log_dir}/{path}_{self.step_tracker}.png')

    def make_timeseries(self, path, data):
        x_ticks = np.linspace(0, self.step_tracker, len(data))
        data = np.array(data)
        fig = plt.figure()
        plt.plot(x_ticks, data, marker='o', linestyle='-')
        plt.xlabel('epoch')
        plt.ylabel('% of dead neurons')
        plt.title('plot of dead neurons over time')
        plt.savefig(f'{self.log_dir}/{path}_{self.step_tracker}.png')
        plt.close(fig)

    def add_to_dict(self, data, path, value, running_average=False):
        if running_average:
            if path not in data:
                data[path] = 0
            else:
                data[path] = self.moving_average(data[path], value)
        else:
            if path not in data:
                data[path] = [value]
            else:
                data[path].append(value)

    def moving_average(self, old_average, new_sample):
        return old_average + (new_sample - old_average) / self.n

    def track_grads(self, grads):
        self.n += 1
        flattened, _ = jax.tree_util.tree_flatten_with_path(grads)
        for key_path, value in flattened:
            value = value.reshape(-1)
            key_path = ".".join(str(k.key) for k in key_path[:-1])
            self.add_to_dict(self.data, key_path, value, running_average=True)

        if self.step_tracker % self.track_every == 0:
            for key_path, value in self.data.items():
                pass

    def track_effective_rank(self, features, sigma=0.1):
        for key, value in features.items():
            s_vals = jnp.linalg.svd(value, compute_uv=False)
            jnp.sort(s_vals, axis=0, descending=True)
            s_vals = s_vals / jnp.sum(s_vals)
            cumulative_s_vals = jnp.cumsum(s_vals)
            s_rank = jnp.argmax(cumulative_s_vals >= 1 - sigma)
            self.add_to_dict(self.timeseries_data['srank'], key, s_rank)

    def step(self, grads, features):
        self.step_tracker += 1
        self.track_grads(grads)
        self.track_effective_rank(features)

    def clear_all(self):
        self.data = {}
        self.timeseries_data = {{'grads': {}, 'srank': {}}}

    def plot_all(self):
        for name, data in self.timeseries_data.items():
            for key_path, values in data.items():
                self.make_timeseries(key_path, values)

