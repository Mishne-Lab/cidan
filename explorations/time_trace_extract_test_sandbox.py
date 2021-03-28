import pickle

import numpy as np
from scipy.io import loadmat

from cidan.LSSC.functions.data_manipulation import reshape_to_2d_over_time
from cidan.LSSC.functions.data_manipulation import save_image
from cidan.LSSC.functions.eigen import generateEigenVectors
from cidan.LSSC.functions.embeddings import calcAffinityMatrix
from cidan.TimeTrace.spatial_temporal_denoising import calculate_spatial_time_denoising
from plotting_functions import time_graph

"""
--input_file /Users/sschickler/Code_Devel/HigleyData/File1_CPn_l5_gcamp6s_lan.tif
--eigen_norm_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File1_CPn_l5_gcamp6s_lan.tif300/embedding_norm_images/embedding_norm_image.png 
--output test.pickle
--rois /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File1_CPn_l5_gcamp6s_lan.tif300/roi_list.json
"""


def extract_time_trace():
    data = loadmat(
        "/Users/sschickler/Code_Devel/LSSC-python/explorations/spatial_test_files/data.mat")[
        "data"].transpose([2, 0, 1])
    Ca_traces_true = loadmat(
        "/Users/sschickler/Code_Devel/LSSC-python/explorations/spatial_test_files/Ca.mat")[
        "Ca"]
    roi_profiles = loadmat(
        "/Users/sschickler/Code_Devel/LSSC-python/explorations/spatial_test_files/profile_set.mat")[
        'profile_set']
    data = data.astype(np.float32)
    data[data < 0] = 0  # TODO ASK ABOUT HOW TO handle data that goes below 0

    # data = data[:300, 10:245, 10: 245]
    # data_2 = reshape_to_2d_over_time(data)

    test = np.zeros((200, 200))
    k = calcAffinityMatrix(
        pixel_list=reshape_to_2d_over_time(data),
        metric="l2",
        knn=50, accuracy=70,
        connections=30,
        normalize_w_k=25,
        num_threads=8,
        spatial_box_num=0,
        temporal_box_num=0,
        total_num_spatial_boxes=1,
        total_num_time_steps=1, save_dir=None,
        progress_signal=None)
    eigen_vectors = generateEigenVectors(K=k,
                                         num_eig=16,
                                         accuracy=10 ** (
                                                 -1 * 6)
                                         ).compute()
    e_vectors_squared = np.power(eigen_vectors, 2)
    e_vectors_sum = np.sum(e_vectors_squared, axis=1)
    e_vectors_sum = np.power(e_vectors_sum, .5)
    e_vectors_sum_rescaled = e_vectors_sum * (
            1.0 / np.percentile(e_vectors_sum, 99))
    e_vectors_sum_rescaled[e_vectors_sum_rescaled > 1.0] = 1.0
    # e_vectors_sum_rescaled[e_vectors_sum_rescaled>255]=255.0
    e_vectors_sum_rescaled = e_vectors_sum_rescaled.reshape([200, 200])

    rois_processed = []
    for x in range(roi_profiles.shape[2] - 1):
        rois_processed.append(
            list(zip(*[list(y) for y in np.nonzero(roi_profiles[:, :, x])])))
    for num, pixels in enumerate(rois_processed):
        for x in pixels:
            test[x[0], x[1]] = 1
    save_image(test, "rois_raw.png")

    rois_processed_1d = [[x[1] + x[0] * 50 for x in roi] for roi in rois_processed]
    time_traces = calculate_spatial_time_denoising(data, rois_processed,
                                                   # calculate_neuropil((200,200),
                                                   #                    rois_processed_1d,
                                                   #                    test.reshape(
                                                   #                        (-1)),
                                                   #                    neuropil_boundary=0),
                                                   e_vectors_sum_rescaled)

    with open("test.pickle", "wb") as file:
        pickle.dump({"spatialtime": time_traces, "true": Ca_traces_true.transpose(),
                     "compare": np.vstack(
                         [time_traces[1], Ca_traces_true.transpose()[1],
                          Ca_traces_true.transpose()[1] - time_traces[1]])}, file,
                    protocol=4)
    # print(np.max(np.abs(time_traces-Ca_traces_true.transpose()), axis=1))
    time_graph.create_graph("test.pickle", "spatial_time_8.png", "")


if __name__ == '__main__':
    for _ in range(10):
        extract_time_trace()
