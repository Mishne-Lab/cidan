import json
import os
import pickle

import fire
import matplotlib.pyplot as plt
import neurofinder
import numpy as np


def graph_psnr(time_traces, true_rois, input_folder):
    with open(true_rois, "r") as json_true:
        rois_true = json.load(json_true)
    time_detected = [0 for _ in range(len(rois_true))]
    number_runs = 0
    inputs = [os.path.join(input_folder, o) for o in os.listdir(input_folder)
              if os.path.isdir(os.path.join(input_folder, o)) and "File1" in o]
    for num, path in enumerate(inputs):
        num_rois_a, num_rois_b = 0, 0
        if os.path.isfile(os.path.join(path, "roi_list.json")):

            try:
                with open(os.path.join(path, "roi_list.json"), "r") as json_b:
                    json_b_actual = json.load(json_b)
                    for roi in json_b_actual:
                        roi["coordinates"] = [[x[0] + 1, x[1] + 1] for x in
                                              roi["coordinates"]]

                a = neurofinder.load(json.dumps(rois_true))
                b = neurofinder.load(json.dumps(json_b_actual))
                matches = neurofinder.match(a, b, threshold=10)
                for num, x in enumerate(matches):
                    if x >= 0:
                        time_detected[num] += 1
                number_runs += 1

            except IndexError:
                print("Error")
        else:
            print("Error")
    with open(time_traces, "rb") as file:

        time_traces_loaded = pickle.load(file)
    std = np.std(time_traces_loaded, axis=1)
    min = np.min(time_traces_loaded, axis=1)
    max = np.max(time_traces_loaded, axis=1)
    time_detected_array = np.array(time_detected, dtype=float) / number_runs
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].scatter(time_detected, std)
    axs[0, 0].set_title('STD')
    axs[1, 0].scatter(time_detected, max)
    axs[1, 0].set_title('Max')
    axs[2, 0].scatter(time_detected, min)
    axs[2, 0].set_title('Min')
    plt.show()


if __name__ == '__main__':
    fire.Fire(graph_psnr)
