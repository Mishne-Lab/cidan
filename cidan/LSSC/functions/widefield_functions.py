import numpy as np


def mask_data_3d(data, mask):
    from cidan.LSSC.functions.data_manipulation import reshape_to_2d_over_time

    """
    Masks a dataset returning a 2d representation with only relevant pixels

    Parameters
    ----------
    data
        A 3d dataset with x,y time
    mask
        A list of pixels to mask in a 1d representation of the pixels
    Returns
    -------
        A 2d data format with pixel num and time as axis
    """
    if len(mask.shape) == 2:
        mask = mask.flatten()
    data_2d = reshape_to_2d_over_time(data)
    data_2d_masked = data_2d[mask]
    return data_2d_masked


def mask_data_2d(data, mask):
    """
    Masks a dataset returning a 2d representation with only relevant pixels

    Parameters
    ----------
    data
        A 2d dataset with pixel num, time
    mask
        A list of pixels to mask in a 1d representation of the pixels
    Returns
    -------
        A 2d data format with pixel num and time as axis
    """
    if len(mask.shape) == 2:
        mask = mask.flatten()
    data_2d_masked = data[mask]
    return data_2d_masked


def mask_to_data_2d(data_masked, mask):
    """
    Takes a masked data and transforms it back into the original
    Parameters
    ----------
    data_masked 1d or 2d data to reformat
    mask 1d mask
    original_shape shape of the 2d data

    Returns
    -------

    """
    if len(mask.shape) == 2:
        mask = mask.flatten()
    if len(data_masked.shape) == 1:
        data_masked = data_masked.reshape((-1, 1))
    num_rows = data_masked.shape[1]
    data_all = np.zeros((mask.shape[0], num_rows))
    data_all = data_all.reshape((-1))
    data_masked = data_masked.reshape((-1))
    mask_two = mask.reshape((-1)).repeat(num_rows)
    data_all[mask_two] = data_masked
    data_all = data_all.reshape((mask.shape[0], num_rows))
    return data_all


def mask_to_data_point(points_list_masked, mask, original_shape):
    """
    Takes a list of points in the masked dataset and transforms them to point numbers in
     original dataset
    Parameters
    ----------
    points_list_masked
    mask
    original_shape

    Returns
    -------

    """
    if len(mask.shape) == 2:
        mask = mask.flatten()
    mask_sum = np.cumsum(mask)

    def get_index(point_num):
        return np.nonzero(mask_sum == point_num + 1)[0][0]

    return [get_index(point_num=point) for point in points_list_masked]


def orig_to_mask_data_point(points_list, mask, original_shape):
    """
    Takes a list of points in the masked dataset and transforms them to point numbers in
     original dataset
    Parameters
    ----------
    points_list_masked
    mask
    original_shape

    Returns
    -------

    """
    if len(mask.shape) == 2:
        mask = mask.flatten()
    mask_sum = np.cumsum(mask)

    return mask_sum[points_list] - 1
    # return [mask_sum[point]-1 for point in points_list_masked]
