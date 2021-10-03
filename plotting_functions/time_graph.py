import pickle

import fire
import matplotlib.pyplot as plt
import numpy as np


def create_graph(time_trace_path, out, roi_select="0 1 2 3 4 5 6 7 8 9 10 11"):
    with open(time_trace_path, "rb") as f:
        time_traces = pickle.load(f)
    std_past = None
    for num_1, key in enumerate(time_traces.keys()):
        mean_flor_den = time_traces[key]
        try:
            if roi_select != "":
                mean_flor_den = [mean_flor_den[int(x)] for x in roi_select.split(" ")]
        except IndexError:
            print("Invalid rois numbers")
        if type(mean_flor_den[0]) == list:
            mean_flor_den = [np.hstack(x) for x in mean_flor_den]
        time_stack = np.vstack(mean_flor_den)
        # time_stack = scipy.ndimage.median_filter(time_stack, (1,15))
        # time_stack = np.divide(time_stack-np.percentile(time_stack,2,axis=1,keepdims=True),(np.percentile(time_stack,98,axis=1,keepdims=True)-np.percentile(time_stack,2,axis=1,keepdims=True)))
        # time_stack[time_stack>1]=1
        # time_stack[time_stack<0] =0
        if std_past is None:
            std = 6 * np.std(time_stack, axis=1)
            std_past = std
        else:
            std = std_past
        time_stack_scaled = time_stack / std.reshape([std.shape[0], 1])
        time_traces_scaled = list(time_stack_scaled)
        x_var = np.arange(1, time_stack_scaled.shape[1] + 1)
        plt.figure(0)
        fig = plt.gcf()
        fig.set_size_inches(10, 6)
        for num, x in enumerate(time_traces_scaled):
            # print((x + num).min(), (x + num).max(), np.mean(x))
            plt.plot(x_var, x - x.mean() + num, color="black" if num_1 != 0 else "red",
                     linewidth=1)
        plt.savefig(out[:-4] + key.replace(" ", "_") + ".png")


if __name__ == '__main__':
    fire.Fire(create_graph)
