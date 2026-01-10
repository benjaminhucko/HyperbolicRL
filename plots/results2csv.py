import csv

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

from pathlib import Path

import numpy as np

def extract_numpy(path):
    ret = {}
    for file in path.rglob("*.np*"):
        data = np.load(file)
        if not isinstance(data, np.ndarray):
            data = data[data.files[0]]

        flattened_data = np.reshape(data, -1)
        ret[file.stem] = flattened_data
    return ret

def extract_tensorboard(log_dir, category):
    accumulator = event_accumulator.EventAccumulator(str(log_dir))
    accumulator.Reload()
    print(accumulator.Tags()["scalars"])

    if category == 'dynamic':
        tags = {"returns": "train/episode_return"}
    elif category == 'stationary':
        tags = {"combined": "train/loss",
                "policy": "train/policy_loss",
                "value": "train/value_loss",
                "entropy": "train/entropy_loss"}
    else:
        raise ValueError(f"{category} is not supported")

    ret = {}
    for tag_key, tag in tags.items():
        events = accumulator.Scalars(tag)
        values = [e.value for e in events]
        ret[tag_key] = np.array(values)
        if 'steps' not in ret:
            ret['steps'] = np.array([e.step for e in events])

    return ret

def make_header(data, long=False):
    if long:
        return ['environment', 'geometry', 'seed'] + list(data.keys())

    data_size = list(data.values())[0].shape[0]
    header = [idx for idx in range(data_size)]
    header = ['environment', 'geometry', 'seed', 'file'] + header
    return header

def traverse_category(base_dir, category, tensorboard=False):
    header = None
    rows = []
    offset = len(base_dir.parts)

    for path in base_dir.rglob(category):
        categories = path.parts
        env_name = categories[offset]
        geometry = categories[offset + 1]
        seed = categories[offset + 2]

        if tensorboard:
            data = extract_tensorboard(path, category)
            if header is None:
                header = make_header(data, long=True)
                rows.append(header)

            for vals in zip(*list(data.values())):
                row = (env_name, geometry, seed) + vals
                rows.append(row)
        else:
            data = extract_numpy(path)
            if header is None:
                header = make_header(data)
                rows.append(header)

            for key, sub_data in data.items():
                row = [env_name, geometry, seed, key] + sub_data.tolist()
                rows.append(row)

    with open(f"data/{category}.csv", "w", newline="") as f:
        writer = csv.writer(f)  # create a writer object
        writer.writerows(rows)

def main():
    base_dir = Path('../logs_paper')
    traverse_category(base_dir, 'eval')
    # traverse_category(base_dir, 'heatmaps')
    # traverse_category(base_dir, 'plots')
    traverse_category(base_dir, 'dynamic', tensorboard=True)
    # traverse_category(base_dir, 'stationary',R tensorboard=True)



if __name__ == '__main__':
    main()