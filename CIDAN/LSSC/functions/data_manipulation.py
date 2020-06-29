import os
from typing import Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from dask import delayed
from scipy import ndimage
# from IPython.display import display, Image
from sklearn.decomposition import PCA

from CIDAN.LSSC.functions.roi_extraction import elbow_threshold


def load_filter_tif_stack(*, path, filter: bool, median_filter: bool,
                          median_filter_size: Tuple[int, int, int],
                          z_score: bool, slice_stack: bool,
                          slice_every, slice_start: int, crop_stack: bool,
                          crop_x: List[int], crop_y: List[int],
                          load_into_mem: bool = True, trial_split=False,
                          trial_split_length=100, trial_num=0):
    """
    This function reads a tiff stack file
    Parameters
    ----------
    path The path to a single tif stack or to a directory of tiff stacks


    Returns
    -------
    a 3D numpy array with the tiff files together or a zarr array if not loading into memory

    """
    if os.path.isdir(path):
        volumes = []
        paths = path if type(path) == list else sorted(os.listdir(path))
        if trial_split:
            paths = paths[trial_num * trial_split_length:(
                                                                 trial_num + 1) * trial_split_length if ( trial_num + 1) * trial_split_length < len(
                paths) else len(paths)]
        for num, x in enumerate(paths):
            file_path = x if type(path) == list else os.path.join(path, x)
            image = tifffile.imread(file_path)
            if len(image.shape) == 2:
                image = image.reshape((1, image.shape[0], image.shape[1]))
            if slice_stack:
                image = image[slice_start::slice_every]
            if crop_stack:
                image = image[:, crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]]
            if filter:
                image = filter_stack(stack=image, median_filter=median_filter,
                                     median_filter_size=median_filter_size,
                                     z_score=z_score)
            volumes.append(image)
            # print("Loading: " + x)

        image = np.vstack(volumes)
        del volumes
        image = image.astype(np.float32)
    elif os.path.isfile(path):
        # return ScanImageTiffReader(path).data()
        image = tifffile.imread(path)
        if len(image.shape) == 2:
            image = image.reshape((1, image.shape[0], image.shape[1]))
        if slice_stack:
            image = image[slice_start::slice_every]
        if crop_stack:
            image = image[:, crop_x[0]:crop_x[1], crop_y[0]:crop_y[1]]
        if filter:
            image = filter_stack(stack=image, median_filter=median_filter,
                                 median_filter_size=median_filter_size, z_score=z_score)
        image = image.astype(np.float32)
    else:
        raise Exception("Invalid Inputs ")
    return image
    # zarr_array = zarr.open('data/example.zarr', mode='w', shape=(10000, 10000),
    #          chunks=(1000, 1000), dtype='i4')

    raise Exception("Invalid Inputs ")
    # vol=ScanImageTiffReader(file_path).data()


def reshape_to_2d_over_time(volume):
    """
    Takes a 3d numpy volume with dim 1 and 2 being x, y and dim 0 being time
    and return array compressing dim 1 and 2 into 1 dimension

    Parameters
    ----------
    volume input 3d numpy volume

    Returns
    -------
    a 2d volume with the 0 dim being pixel number and 1 being time value

    """
    return np.transpose(np.reshape(volume, (volume.shape[0], -1), order="C"))


def save_image(volume: np.ndarray, name: str, directory: str, shape: tuple,
               number_save: int):
    """
    Function to save images from a 3d volume
    Parameters
    ----------
    volume
    name
    directory
    shape shape to reshape volume into
    number_save number of slices to take and save

    Returns
    -------
    None
    """
    for x in range(number_save):
        fig, ax = plt.subplots()
        imgplot = ax.imshow(np.reshape(volume,
                                       shape)[shape[0] // number_save * x])
        fig.savefig(os.path.join(directory, name + "_" + str(x)))

        # img = Image.fromarray(
        #     np.reshape(volume,
        #                shape)[shape[0]/number_save*x] * 255).convert('L')
        # img.save(
        #     os.path.join(directory,name+"_"+str(x)))


def filter_stack(*, stack: np.ndarray, median_filter: bool,
                 median_filter_size: Tuple[int, int, int],
                 z_score: bool, hist_eq=False):
    if median_filter and not hist_eq:
        stack = ndimage.median_filter(stack, median_filter_size)
        # stack = ndimage.filters.convolve(stack, np.full((3, 3, 3), 1.0 / 27))
    if z_score and not hist_eq:
        stack_t = np.transpose(stack, (1, 2, 0))
        shape = (stack.shape[1], stack.shape[2], 1)
        std = np.std(stack_t, axis=2).reshape(shape)
        mean = np.mean(stack_t, axis=2).reshape(shape)
        stack_t = (stack_t - mean) / std
        stack = np.transpose(stack_t, (2, 0, 1))
    if hist_eq:
        stack_t = np.transpose(stack, (1, 2, 0))
        quant_5 = np.percentile(stack_t, 5, axis=2)
        quant_95 = np.percentile(stack_t, 95, axis=2)
        quant_5_filtered = ndimage.filters.gaussian_filter(quant_5, 2)
        quant_95_filtered = ndimage.filters.gaussian_filter(quant_95, 2)
        stack_equalized = np.divide(
            np.subtract(stack_t, quant_5_filtered[..., np.newaxis]), (
                    quant_95_filtered - quant_5_filtered + .00000001)[..., np.newaxis])

        stack_equalized = np.nan_to_num(stack_equalized)

        stack_equalized[stack_equalized > 1] = 1
        stack_equalized[stack_equalized < 0] = 0
        stack_equalized_squared = np.power(stack_equalized, 2)
        stack_equalized_filtered = ndimage.median_filter(stack_equalized_squared,
                                                         (1, 1, 3))
        stack = np.transpose(stack_equalized_filtered, (2, 0, 1))

    return stack


def pixel_num_to_2d_cord(pixel_list, volume_shape: Tuple[int, int]):
    """
    Takes a list of pixels and converts it to a list of cords for each pixel

    Parameters
    ----------
    pixel_list : np.ndarray
        list of pixel numbers
    volume_shape : tuple
        shape of the original dataset [x,y]

    Returns
    -------
    a list of tuples where each is the cordinates for each pixel

    """

    def convert_one(num):
        return num % volume_shape[1], num // volume_shape[1]

    return np.apply_along_axis(convert_one, axis=0, arr=pixel_list)


@delayed
def join_data_list(data_list):
    """
    Joins a list of trials with the same dimensions
    Parameters
    ----------
    data_list
        Data to stack

    Returns
    -------
    List of data stacked over 1st dimension
    """
    return np.vstack(data_list)


@delayed
def saveTempImage(data, save_dir, spatial_box_num):
    # print(save_dir)
    temp_dir = os.path.join(save_dir, "temp_images")
    # print(data[60, :, :].shape)
    img = Image.fromarray(data[3, :, :] * 255).convert('L')
    image_path = os.path.join(temp_dir, "embedding_norm_image_box_{}.png".format(
        spatial_box_num))
    img.save(image_path)
    return data


def applyPCA(data, pca_threshold):
    pca = PCA(n_components=pca_threshold)
    pca.fit(reshape_to_2d_over_time(data).transpose())
    threshold = elbow_threshold(pca.explained_variance_,
                                np.arange(0, len(pca.explained_variance_ratio_), 1),
                                half=False) * .8
    filtered_pca_components = pca.components_[pca.explained_variance_ > threshold]
    return filtered_pca_components.reshape((-1, data.shape[1], data.shape[2]))
