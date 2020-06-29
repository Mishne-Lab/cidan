import dask
import numpy as np
from skimage import feature
from skimage.transform import rescale, hough_circle, hough_circle_peaks

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
    if len(roi_list)==0:
        return []
    out_list = [filterRoi(x, image_shape=image_shape, threshold=threshold) for x in
                roi_list]
    out = dask.compute(*out_list)
    # return [int(x[0]) for x in out]
    return [int(x) for x in out / max(out) * 100]


# @dask.delayed
def filterRoi(roi, image_shape, threshold):
    roi_2d_cord = pixel_num_to_2d_cord(roi, volume_shape=image_shape)
    roi_max_cord = np.max(roi_2d_cord, axis=1)
    roi_min_cord = np.min(roi_2d_cord, axis=1)
    newShape = roi_max_cord - roi_min_cord + np.array([10, 10])
    image = np.zeros(newShape)
    for cord in zip(
            *list(roi_2d_cord - np.reshape(roi_min_cord - np.array([2, 2]), (2, 1)))):
        image[cord[0], cord[1]] = 1

    image = rescale(image, 2, anti_aliasing=False)
    image_canny = feature.canny(image)
    # radii =range(int(np.min(roi_max_cord - roi_min_cord)*.8),math.ceil(np.min(roi_max_cord - roi_min_cord)*1.2)+1,1)
    radii = [np.min(roi_max_cord - roi_min_cord)]
    result = hough_circle_peaks(hough_circle(image_canny, radii), radii,
                                total_num_peaks=1)
    # result = hough(image_canny, accuracy=20, threshold=0,
    #                        min_size=int(np.min(roi_max_cord - roi_min_cord)/1.7), max_size=int(np.max(roi_max_cord - roi_min_cord)*2.1))

    return result[0][0]
    result.sort(order='accumulator')
    result_filtered = filter(lambda x: not any([y == 0 for y in x]), reversed(result))
    sqrt_2 = 2 ** .5
    for x in result_filtered:
        if (int(np.max([x[3], x[4]])) <= np.max(roi_max_cord - roi_min_cord) * sqrt_2):
            if (np.min([x[3], x[4]]) >= np.min(roi_max_cord - roi_min_cord)):
                return list(x) + [(roi_max_cord - roi_min_cord)]

    return [0.01]
    surface_pixels = np.count_nonzero(image_canny)
    return ((((np.min(roi_max_cord - roi_min_cord)) * 3.14))) / (surface_pixels / 4)
