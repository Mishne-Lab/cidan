import os
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
import tifffile
from LSSC.Parameters import Parameters
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Union, Any, List, Optional, cast, Tuple, Dict

# from IPython.display import display, Image
from dask import delayed

def load_filter_tif_stack(*, path, filter: bool, median_filter: bool,
                          median_filter_size: Tuple[int, int,int],
                          z_score: bool, slice_stack: bool,
                          slice_every, slice_start: int):
    """
    This function reads a tiff stack file
    Parameters
    ----------
    path The path to a single tif stack

    Returns
    -------
    a 3D numpy array with the tiff files together

    """
    if os.path.isdir(path):
        volumes = []
        for num, x in enumerate(sorted(os.listdir(path))):
            file_path = os.path.join(path, x)
            image = tifffile.imread(file_path)
            if slice_stack:
                image = image[slice_start::slice_every, :, :]
            if filter:
                image = filter_stack(stack=image, median_filter=median_filter,
                                     median_filter_size=median_filter_size,
                                     z_score=z_score)
            volumes.append(image)
            print("Loading: "+x)

        image = np.vstack(volumes)
        return image
    if os.path.isfile(path):
        # return ScanImageTiffReader(path).data()
        image = tifffile.imread(path)

        if slice_stack:
            image = image[slice_start::slice_every,:,:]
        if filter:
            image = filter_stack(stack=image, median_filter=median_filter,
                                 median_filter_size=median_filter_size, z_score=z_score)
        return image
    raise Exception("Invalid Input folders not allowed currently ")
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
    return np.transpose(np.reshape(volume, (volume.shape[0],-1), order="C"))
def save_image(volume: np.ndarray, name: str, directory: str, shape: tuple, number_save: int):
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
                       shape)[shape[0]//number_save*x])
        fig.savefig(os.path.join(directory,name+"_"+str(x)))

        # img = Image.fromarray(
        #     np.reshape(volume,
        #                shape)[shape[0]/number_save*x] * 255).convert('L')
        # img.save(
        #     os.path.join(directory,name+"_"+str(x)))
def filter_stack(*,stack: np.ndarray, median_filter: bool,
                 median_filter_size: Tuple[int, int, int],
                 z_score: bool,):

    if z_score:
        stack_t  = np.transpose(stack, (1, 2, 0))
        shape = (1,stack_t.shape[1],stack_t.shape[2])
        std = np.std(stack_t,axis=0).reshape(shape)
        mean = np.mean(stack_t,axis=0).reshape(shape)
        stack_t = (stack_t-mean)/std
        stack = np.transpose(stack_t, (2, 0, 1))
    if median_filter:
        stack = ndimage.median_filter(stack, median_filter_size)
    return stack
