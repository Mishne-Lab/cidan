import numpy as np

from CIDAN.TimeTrace.mean import calculateMeanTrace


def calculateDeltaFOverF(roi_data, neuropil, roi_denoised_data, denoise=True,
                         sub_neuropil=False):
    """
    Calcualtes the DeltaF Over F time trace for a given roi and data
    Parameters
    ----------
    roi : np.ndarray or list
        list of pixels for data (1D cords)
    data : np.ndarray
        2D list of data for a single trials with pixel num as dim 0, time as dim 1
    denoise : bool
        Whether to denoise mean floresence of data using wavelets
    Returns
    -------
    A 1d nd array with the time trace for the ROI
    """

    time_trace = calculateMeanTrace(roi_data, neuropil, roi_denoised_data, denoise,
                                    sub_neuropil)

    bottom_10_number = np.percentile(time_trace, 10)

    bottom_10_frames = time_trace[time_trace < bottom_10_number]
    bottom_10_avg = np.mean(bottom_10_frames)

    if bottom_10_avg == 0:
        return time_trace
    return (time_trace - bottom_10_avg) / bottom_10_avg
