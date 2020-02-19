import os
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader
import tifffile
from LSSC.Parameters import Parameters
import matplotlib.pyplot as plt
from scipy import ndimage
# from IPython.display import display, Image
def load_tif_stack(path):
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
        raise Exception( "Invalid Input folders not allowed currently ")
        # for num, x in enumerate(os.listdir(path)):
        #     file_path = os.path.join(path, x)
        #     if num == 0:
        #         vol = ScanImageTiffReader(file_path).data()
        #         print(vol)
    if os.path.isfile(path):
        # return ScanImageTiffReader(path).data()
        return tifffile.imread(path)
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
        imgplot = plt.imshow(np.reshape(volume,
                       shape)[shape[0]//number_save*x])
        plt.savefig(os.path.join(directory,name+"_"+str(x)))

        # img = Image.fromarray(
        #     np.reshape(volume,
        #                shape)[shape[0]/number_save*x] * 255).convert('L')
        # img.save(
        #     os.path.join(directory,name+"_"+str(x)))
def filter_stack(stack: np.ndarray, parameters: Parameters):
    if parameters.z_score:
        stack_t = np.transpose(stack,(1,2,0))
        std = np.std(stack_t,axis=2)
        mean = np.mean(stack_t,axis=2)
        stack = (stack_t-mean)/std
    if parameters.median_filter:
        stack = ndimage.median_filter(stack, parameters.median_filter_size)
    return stack
