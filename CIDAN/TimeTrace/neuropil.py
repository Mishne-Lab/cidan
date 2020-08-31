import numpy as np
from scipy.ndimage import binary_dilation


def calculate_nueropil(image_shape, roi_list, roi_mask_flat, min_pixels=25):
    neuropil = []
    for roi in roi_list:
        original_size = len(roi)
        roi_mask = np.reshape(roi_mask_flat, image_shape).copy()
        roi_mask[roi_mask > 0] = 1
        roi_opposite_mask = (roi_mask == 0)
        roi_image = np.zeros(roi_mask_flat.shape)
        roi_image[roi] = 1

        roi_image = np.reshape(roi_image, image_shape)
        roi_image_original = roi_image.copy()
        while np.count_nonzero(roi_image - roi_mask > 0) - original_size < min_pixels:
            roi_image = binary_dilation(input=roi_image)
        neuropil.append(np.nonzero(np.reshape(roi_image - roi_mask, (-1))))
    return neuropil
