import pickle

import fire
import matplotlib.pyplot as plt
import numpy as np


def create_graph(time_trace_path, out):
    with open(time_trace_path, "rb") as f:
        time_traces = pickle.load(f)
    for num, key in enumerate(time_traces.keys()):
        mean_flor_den = time_traces[key]
        time_stack = np.vstack([np.hstack(x) for x in mean_flor_den])[:10, :]
        # time_stack = scipy.ndimage.median_filter(time_stack, (1,15))
        std = np.std(time_stack, axis=1)
        time_stack_scaled = time_stack / (6 * np.max(std))
        time_traces_scaled = list(time_stack_scaled)
        x_var = np.arange(1, time_stack_scaled.shape[1] + 1)
        plt.figure(num)
        for num, x in enumerate(time_traces_scaled):
            print((x + num).min(), (x + num).max(), np.mean(x))
            plt.plot(x_var, x + num - np.mean(x), color="black", linewidth=.5)
        plt.savefig(out[:-4] + key.replace(" ", "_") + ".png")


if __name__ == '__main__':
    fire.Fire(create_graph)
