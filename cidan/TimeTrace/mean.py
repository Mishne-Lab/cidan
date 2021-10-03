import numpy as np


def calculateMeanTrace(roi_data, neuropil_data,
                       sub_neuropil=False):
    """
    Calcualtes the mean time trace for a given roi and data
    Parameters
    ----------
    roi list of pixels for data (1D cords)
    data 2D list of data for a single trials with pixel num as dim 0, time as dim 1

    Returns
    -------
    A 1d nd array with the time trace for the ROI
    """

    mean = np.mean(roi_data, axis=0)
    if sub_neuropil:
        mean = mean - np.mean(neuropil_data, axis=0)
    # if denoise:
    #     mean = waveletDenoise(mean)
    return mean


def neuropil(roi_data, neuropil_data, ):
    return np.mean(neuropil_data, axis=0)
