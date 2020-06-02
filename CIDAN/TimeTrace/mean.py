import numpy as np


def calculateMeanTrace(roi, data):
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
    return np.mean(data[roi], axis=0)
