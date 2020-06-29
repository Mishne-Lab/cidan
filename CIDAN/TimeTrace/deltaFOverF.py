import numpy as np

from CIDAN.TimeTrace.waveletDenoise import waveletDenoise


def calculateDeltaFOverF(roi, data, denoise=True):
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
    pixels_in_roi = data[roi]
    pixel_sum = np.sum(pixels_in_roi, axis=1)
    bottom_10_number = np.percentile(pixel_sum, 10)
    bottom_10_pixels = pixels_in_roi[pixel_sum < bottom_10_number]
    bottom_10_avg = np.mean(bottom_10_pixels, axis=0)
    mean = np.mean(pixels_in_roi, axis=0)
    if denoise:
        denoised_mean = waveletDenoise(mean.reshape((1, -1)))
        mean = denoised_mean.reshape((-1))
    return (mean - bottom_10_avg) / bottom_10_avg
