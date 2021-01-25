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
    data_2d_masked = data[mask]
    return data_2d_masked


def mask_to_data_2d(data_masked, mask, original_shape):
    """
    Takes a masked data and transforms it back into the original
    Parameters
    ----------
    data_masked
    mask
    original_shape

    Returns
    -------

    """


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
