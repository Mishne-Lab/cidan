from functools import reduce

import numpy as np
from scipy import optimize

from cidan.LSSC.functions.data_manipulation import reshape_to_2d_over_time


# TODO Scale A to 0-1
def calculate_spatial_time_denoising(dataset, rois, neuropil, eigen_norm_image):
    roi_images = np.zeros([eigen_norm_image.shape[0], eigen_norm_image[1], len(rois)])
    for num, pixels in enumerate(rois):
        for x in pixels:
            roi_images[x[0], x[1], num] = 1
    A = np.dstack([eigen_norm_image for _ in rois])
    A[roi_images == 0] = 0
    roi_mask_neuropil = np.zeros([eigen_norm_image.shape[0], eigen_norm_image[1]])
    neuropil_all = reduce(lambda x, y: x + y, neuropil)
    roi_mask_neuropil[neuropil_all] = 1
    dataset_masked = np.copy(dataset)
    dataset_masked[roi_mask_neuropil] = 0

    _, _, vt = np.linalg.svd(reshape_to_2d_over_time(dataset_masked),
                             full_matrices=False)
    C_b = vt.transpose()[:, 0]
    dataset_2d = reshape_to_2d_over_time(dataset)
    A_b, _ = optimize.nnls(dataset_2d, C_b)
    C = optimize.nnls(dataset_2d - A_b * C_b, A)
    return C
