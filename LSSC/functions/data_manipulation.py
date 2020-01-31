import os
import numpy as np
from ScanImageTiffReader import ScanImageTiffReader

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
        return ScanImageTiffReader(path).data()
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