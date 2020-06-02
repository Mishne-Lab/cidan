import dask
import numpy as np
from skimage import feature

from CIDAN.LSSC.functions.data_manipulation import pixel_num_to_2d_cord


def filterRoiList(roi_list, image_shape, threshold=.7):
    """
    Filters a list of ROIS based on looking at surface area vs surface area of a similar sized circle of each roi
    Parameters
    ----------
    roi_list
    image_shape
    threshold

    Returns
    -------
    list of threshold_values
    """
    out_list = [filterRoi(x, image_shape=image_shape, threshold=threshold) for x in
                roi_list]
    return dask.compute(*out_list)


def filterRoi(roi, image_shape, threshold):
    roi_2d_cord = pixel_num_to_2d_cord(roi, volume_shape=image_shape)
    roi_max_cord = np.max(roi_2d_cord, axis=1)
    roi_min_cord = np.min(roi_2d_cord, axis=1)
    newShape = roi_max_cord - roi_min_cord + np.array([10, 10])
    image = np.zeros(newShape)
    for cord in zip(
            *list(roi_2d_cord - np.reshape(roi_min_cord - np.array([2, 2]), (2, 1)))):
        image[cord[0], cord[1]] = 1
    image_canny = feature.canny(image)
    surface_pixels = np.count_nonzero(image_canny)
    return ((((np.min(roi_max_cord - roi_min_cord)) * 3.14))) / surface_pixels
