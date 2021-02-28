import dask
import numpy as np
from scipy import optimize
from sklearn.decomposition import NMF

from cidan.LSSC.functions.data_manipulation import reshape_to_2d_over_time, save_image


# TODO Scale A to 0-1
def calculate_spatial_time_denoising(dataset, rois, neuropil, eigen_norm_image):
    roi_images = np.zeros(
        [len(rois), eigen_norm_image.shape[0], eigen_norm_image.shape[1]], dtype=int)
    for num, pixels in enumerate(rois):
        for x in pixels:
            roi_images[num, x[0], x[1]] = 1
    A = np.vstack([eigen_norm_image.reshape(
        [1, eigen_norm_image.shape[0], eigen_norm_image.shape[1]]) for _ in rois])
    save_image(A.sum(axis=0), "A.png")
    A[roi_images == 0] = 0
    roi_mask_neuropil = np.zeros(
        [eigen_norm_image.shape[0] * eigen_norm_image.shape[1]], dtype=int)

    neuropil_all = np.hstack([x[0] for x in neuropil])
    roi_mask_neuropil[neuropil_all] = 1

    roi_mask_neuropil = roi_mask_neuropil.reshape(
        [eigen_norm_image.shape[0], eigen_norm_image.shape[1]])
    roi_mask_neuropil += np.sum(roi_images, axis=0)
    save_image(np.sum(roi_images, axis=0), "rois.png")
    save_image(roi_mask_neuropil, "roi_mask_neuropil.png")
    save_image(np.sum(dataset, axis=0), "dataset.png")
    dataset_masked = np.copy(dataset).transpose([1, 2, 0])

    dataset_masked[roi_mask_neuropil == 1] = 0
    dataset_masked = dataset_masked.transpose([2, 0, 1])
    save_image(np.sum(dataset_masked, axis=0), "dataset_masked.png")
    dataset_masked_r = reshape_to_2d_over_time(dataset_masked)
    model = NMF(n_components=1, init='custom', alpha=0, solver='mu',
                beta_loss='kullback-leibler')
    A_mask = model.fit_transform(dataset_masked_r, 1,
                                 np.mean(dataset_masked_r, axis=1).reshape(-1, 1),
                                 np.mean(dataset_masked_r, axis=0).reshape(1, -1))
    C_b = model.components_
    save_image(A_mask.reshape([eigen_norm_image.shape[0], eigen_norm_image.shape[1]]),
               "A_mask.png")

    dataset_2d = reshape_to_2d_over_time(dataset)
    A_b, t = optimize.nnls(dataset_2d.transpose(), C_b[0, :])
    A_b = np.power(A_b.reshape([eigen_norm_image.shape[0], eigen_norm_image.shape[1]]),
                   -1)
    A_b[A_b == np.inf] = 0
    save_image(A_b, "A_background.png")

    C = dask.compute([dask.delayed(optimize.nnls)(
        dataset_2d - A_b.reshape([-1, 1]) * C_b, reshape_to_2d_over_time(A)[:, x]) for x
                      in range(len(rois))])
    C = np.vstack([x[0] for x in C[0]]).transpose()
    return C
