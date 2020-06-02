import numpy as np
def calculateDeltaFOverF(roi, data):
    """
    Calcualtes the deltaf/f time trace for a given roi and data
    Parameters
    ----------
    roi list of pixels for data (1D cords)
    data 2D list of data for a single trials with pixel num as dim 0, time as dim 1

    Returns
    -------
    A 1d nd array with the time trace for the ROI
    """
    pixels_in_cluster = data[roi]
    pixel_sum = np.sum(pixels_in_cluster, axis=1)
    bottom_10_number = np.percentile(pixel_sum, 10)
    bottom_10_pixels = pixels_in_cluster[pixel_sum < bottom_10_number]
    bottom_10_avg = np.mean(bottom_10_pixels, axis=0)
    return (np.mean(pixels_in_cluster, axis=0) - bottom_10_avg) / bottom_10_avg
