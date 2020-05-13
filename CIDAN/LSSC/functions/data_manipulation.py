import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tifffile
from PIL import Image
from dask import delayed
from scipy import ndimage


# from IPython.display import display, Image

def load_filter_tif_stack(*, path, filter: bool, median_filter: bool,
                          median_filter_size: Tuple[int, int, int],
                          z_score: bool, slice_stack: bool,
                          slice_every, slice_start: int):
    """
    This function reads a tiff stack file
    Parameters
    ----------
    path The path to a single tif stack or a list of paths to many tiff stacks


    Returns
    -------
    a 3D numpy array with the tiff files together

    """

    if type(path) == list or os.path.isdir(path):
        volumes = []
        paths = path if type(path) == list else sorted(os.listdir(path))

        for num, x in enumerate(paths):
            file_path = x if type(path) == list else os.path.join(path, x)
            image = tifffile.imread(file_path)
            if slice_stack:
                image = image[slice_start::slice_every, :, :]
            if filter:
                image = filter_stack(stack=image, median_filter=median_filter,
                                     median_filter_size=median_filter_size,
                                     z_score=z_score)
            volumes.append(image)
            print("Loading: " + x)

        image = np.vstack(volumes)
        del volumes
        return image
    if os.path.isfile(path):
        # return ScanImageTiffReader(path).data()
        image = tifffile.imread(path)

        if slice_stack:
            image = image[slice_start::slice_every, :, :]
        if filter:
            image = filter_stack(stack=image, median_filter=median_filter,
                                 median_filter_size=median_filter_size, z_score=z_score)
        return image
    raise Exception("Invalid Inputs folders not allowed currently ")
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
                 z_score: bool, ):
    if z_score:
        stack_t = np.transpose(stack, (1, 2, 0))
        shape = (1, stack_t.shape[1], stack_t.shape[2])
        std = np.std(stack_t, axis=0).reshape(shape)
        mean = np.mean(stack_t, axis=0).reshape(shape)
        stack_t = (stack_t - mean) / std
        stack = np.transpose(stack_t, (2, 0, 1))
    if median_filter:
        stack = ndimage.median_filter(stack, median_filter_size)
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
    print(save_dir)
    temp_dir = os.path.join(save_dir, "temp_images")
    print(data[60, :, :].shape)
    img = Image.fromarray(data[3, :, :] * 255).convert('L')
    image_path = os.path.join(temp_dir, "embedding_norm_image_box_{}.png".format(
        spatial_box_num))
    img.save(image_path)
    return data
