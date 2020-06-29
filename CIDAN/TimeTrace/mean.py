import numpy as np

from CIDAN.TimeTrace.waveletDenoise import waveletDenoise


def calculateMeanTrace(roi, data, denoise=True):
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
    if denoise:
        return waveletDenoise(np.mean(data[roi], axis=0).reshape((1, -1))).reshape((-1))
    return np.mean(data[roi], axis=0)
