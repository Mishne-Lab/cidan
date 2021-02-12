import json
import pickle

import fire
import numpy as np

from cidan.LSSC.functions.data_manipulation import load_filter_tif_stack, \
    reshape_to_2d_over_time
from cidan.TimeTrace.mean import calculateMeanTrace


def extract_time_trace(input_file, rois, output):
    data = load_filter_tif_stack(path=input_file, filter=False, median_filter=False,
                                 median_filter_size=0, z_score=False, slice_every=0,
                                 slice_stack=False, slice_start=0, crop_stack=False,
                                 crop_y=None, crop_x=None)
    data_2 = reshape_to_2d_over_time(data[1])
    with open(rois, "rb") as file:
        rois = json.load(file)
    time_traces = []
    for roi in rois:
        roi = roi["coordinates"]
        roi_1d = [x[0] + x[1] * data[0][1] for x in roi]

        time_traces.append(
            calculateMeanTrace(data_2[roi_1d], None, denoise=False, sub_neuropil=False))
    with open(output, "wb") as file:
        pickle.dump(np.vstack(time_traces), file, protocol=4)


if __name__ == '__main__':
    fire.Fire(extract_time_trace)
