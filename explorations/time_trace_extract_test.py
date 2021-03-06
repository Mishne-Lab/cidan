import json
import pickle

import fire
import matplotlib.image as mpimg
import numpy as np

from cidan.LSSC.functions.data_manipulation import load_tif_stack
from cidan.TimeTrace.neuropil import calculate_neuropil
from cidan.TimeTrace.spatial_temporal_denoising import calculate_spatial_time_denoising
from plotting_functions import time_graph

"""
--input_file /Users/sschickler/Code_Devel/HigleyData/File1_CPn_l5_gcamp6s_lan.tif
--eigen_norm_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File1_CPn_l5_gcamp6s_lan.tif300/embedding_norm_images/embedding_norm_image.png 
--output test.pickle
--rois /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File1_CPn_l5_gcamp6s_lan.tif300/roi_list.json
"""
def extract_time_trace(input_file, rois, output, eigen_norm_path):
    data = load_tif_stack(path=input_file)
    data = data.astype(np.float32)
    data[data<0]=0
    data = data[:2000, 10:245, 10: 245]
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
    time_traces = calculate_spatial_time_denoising(data, rois_processed,
                                                   calculate_neuropil((235, 235),
                                                                      rois_processed_1d,
                                                                      test.reshape(
                                                                          (-1)),
                                                                      neuropil_boundary=0),
                                                   eigen_norm)

    with open(output, "wb") as file:
        pickle.dump({"spatialtime":np.vstack(time_traces).transpose()}, file, protocol=4)

    time_graph.create_graph(output, "test.png", "0 1 2 3 4 5 6 7 8 9 10" )
if __name__ == '__main__':
    fire.Fire(extract_time_trace)
