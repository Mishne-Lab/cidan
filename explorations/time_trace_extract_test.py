import json
import pickle

import fire
import matplotlib.image as mpimg
import numpy as np

from cidan.LSSC.functions.data_manipulation import load_filter_tif_stack
from cidan.TimeTrace.neuropil import calculate_neuropil
from cidan.TimeTrace.spatial_temporal_denoising import calculate_spatial_time_denoising


def extract_time_trace(input_file, rois, output, eigen_norm_path):
    _, data = load_filter_tif_stack(path=input_file, filter=False, median_filter=False,
                                    median_filter_size=0, z_score=False, slice_every=0,
                                    slice_stack=False, slice_start=0, crop_stack=True,
                                    crop_y=[10, 245], crop_x=[10, 245])
    data = data[:400, :, :]
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
        pickle.dump(np.vstack(time_traces), file, protocol=4)


if __name__ == '__main__':
    fire.Fire(extract_time_trace)
