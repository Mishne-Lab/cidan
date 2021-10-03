import json
import pickle

import fire
import matplotlib.image as mpimg
import numpy as np

from cidan.LSSC.functions.data_manipulation import load_tif_stack, \
    reshape_to_2d_over_time
from cidan.TimeTrace.spatial_temporal_denoising import calculate_spatial_time_denoising
from plotting_functions import time_graph

"""
--input_file /Users/sschickler/Code_Devel/HigleyData/File1_CPn_l5_gcamp6s_lan.tif
--eigen_norm_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File1_CPn_l5_gcamp6s_lan.tif300/embedding_norm_images/embedding_norm_image.png 
--output test.pickle
--rois /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File1_CPn_l5_gcamp6s_lan.tif300/roi_list.json
--input_file /Users/sschickler/Code_Devel/HigleyData/File5_l23_gcamp6s_lan.tif 
--eigen_norm_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan_worse/File5_l23_gcamp6s_lan.tif10021/embedding_norm_images/embedding_norm_image.png 
--output test.pickle
--rois /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan_worse/File5_l23_gcamp6s_lan.tif10021/roi_list.json

"""


def extract_time_trace(input_file, rois, output, eigen_norm_path):
    data = load_tif_stack(path=input_file, convert_to_32=False)
    data = data.astype(float)
    data[data > 2 ** 15 + 512] = 2 ** 15 + 512
    data = data - 2 ** 15
    data[data < 0] = 0
    data = data[500:1500, 10:245, 10: 245]
    # tiffile.imsave("test.tif",data.astype(np.float32))
    # data_2 = reshape_to_2d_over_time(data)
    with open(rois, "rb") as file:
        rois = json.load(file)
    test = np.zeros((235, 235))
    eigen_norm = mpimg.imread(eigen_norm_path)
    rois_processed = [[[x[0] - 10, x[1] - 10] for x in roi["coordinates"] if
                       235 > x[0] - 10 >= 0 and 235 > x[1] - 10 >= 0] for roi in
                      rois]

    for num, pixels in enumerate(rois_processed):
        for x in pixels:
            test[x[0], x[1]] = 1
    rois_processed_1d = [[x[1] + x[0] * 235 for x in roi] for roi in rois_processed]

    time_traces, background = calculate_spatial_time_denoising(data, rois_processed,
                                                               # calculate_neuropil((235, 235),
                                                               #                    rois_processed_1d,
                                                               #                    test.reshape(
                                                               #                        (-1)),
                                                               #                    neuropil_boundary=0),
                                                               eigen_norm)
    roi_traces = []
    for roi in rois_processed_1d:
        roi_traces.append(np.mean(reshape_to_2d_over_time(data)[roi], axis=0))
    select_thing = [42, 43, 96]

    mean = np.vstack([roi_traces[x] for x in select_thing] + [
        np.mean(np.mean(data, axis=1), axis=1).reshape((1, -1))])
    time_traces = time_traces[select_thing]
    time_traces = np.vstack([time_traces, background.reshape((1, -1))])
    print("STD True", mean.std(axis=1))
    print("STD spatial", time_traces.std(axis=1))
    print("Mean True", mean.mean(axis=1))
    print("Mean spatial", time_traces.mean(axis=1))
    import pandas as pd
    test = pd.DataFrame.from_dict(
        {"STD Mean trace": mean.std(axis=1),
         "STD spatial": time_traces.std(axis=1),
         "Mean mean trace": mean.mean(axis=1),
         "Mean spatial": time_traces.mean(axis=1)})
    with open(output, "wb") as file:
        pickle.dump({"spatialtime": time_traces, "mean": mean}, file, protocol=4)

    time_graph.create_graph(output, "File5.png")

if __name__ == '__main__':
    fire.Fire(extract_time_trace)
