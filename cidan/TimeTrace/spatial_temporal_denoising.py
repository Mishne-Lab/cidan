import time

import numpy as np
from scipy import ndimage
from sklearn.decomposition import NMF

from cidan.LSSC.functions.data_manipulation import reshape_to_2d_over_time, save_image


# TODO Scale A to 0-1
def calculate_spatial_time_denoising(dataset, rois, eigen_norm_image):
    time_all = time.time()

    dataset = dataset.astype(np.float32)
    roi_images = np.zeros(
        [len(rois), eigen_norm_image.shape[0], eigen_norm_image.shape[1]], dtype=int)
    for num, pixels in enumerate(rois):
        for x in pixels:
            roi_images[num, x[0], x[1]] = 1
    A = np.vstack([eigen_norm_image.reshape(
        [1, eigen_norm_image.shape[0], eigen_norm_image.shape[1]]) for _ in rois])
    save_image(A.sum(axis=0), "A.png")

    min_vals = np.amin(A.reshape([A.shape[0], -1]), where=roi_images.reshape([A.shape[0], -1]) == 1, initial=1, axis=1)
    max_vals = np.amax(A.reshape([A.shape[0], -1]), where=roi_images.reshape([A.shape[0], -1]) == 1, initial=0, axis=1)
    A = (A - min_vals.reshape((A.shape[0], 1, 1))) / (
                max_vals.reshape((A.shape[0], 1, 1)) - min_vals.reshape((A.shape[0], 1, 1)))
    A[roi_images == 0] = 0
    # roi_mask_neuropil = np.zeros(
    #     [eigen_norm_image.shape[0] * eigen_norm_image.shape[1]], dtype=int)
    #
    # neuropil_all = np.hstack([x[0] for x in neuropil])
    test = np.sum(roi_images, axis=0)

    test[test > 1] = 1
    roi_mask_neuropil = ndimage.morphology.binary_dilation(test,
                                                           iterations=6)  # image_temp = ndimage.morphology.binary_dilation(image_temp)

    roi_mask_neuropil = roi_mask_neuropil.reshape(
        [eigen_norm_image.shape[0], eigen_norm_image.shape[1]])
    save_image(roi_mask_neuropil, "roi_neuropil.png")

    roi_mask_neuropil = roi_mask_neuropil + np.sum(roi_images, axis=0)

    for x in range(roi_images.shape[0]):
        save_image(roi_images[x], "rois/roi%s.png" % str(x))
    save_image(roi_mask_neuropil, "roi_mask_neuropil.png")
    save_image(np.sum(dataset, axis=0), "dataset.png")
    dataset_masked = np.copy(dataset).transpose([1, 2, 0])

    dataset_masked[roi_mask_neuropil >= 1] = 0
    dataset_masked = dataset_masked.transpose([2, 0, 1])
    save_image(np.sum(dataset_masked, axis=0), "dataset_masked.png")
    # print("data_mask", time.time()-time_all)
    dataset_masked_r = reshape_to_2d_over_time(dataset_masked)
    model = NMF(n_components=1, init='custom', alpha=0, solver='mu',
                beta_loss='kullback-leibler')
    A_mask = time_thing(model.fit_transform)(dataset_masked_r, 1,
                                             np.mean(dataset_masked_r, axis=1).reshape(-1, 1),
                                             np.mean(dataset_masked_r, axis=0).reshape(1, -1))
    C_b = model.components_
    save_image(A_mask.reshape([eigen_norm_image.shape[0], eigen_norm_image.shape[1]]),
               "A_background_mask.png")
    # print("A_background_mask",time.time()-time_all)

    dataset_2d = reshape_to_2d_over_time(dataset)

    pixels_to_calculate = np.nonzero(np.sum(reshape_to_2d_over_time(roi_images[:, :, :]), axis=1))[0]
    C, _ = iterative_solve(dataset_2d, A=A, C_b=C_b,
                           pixels_to_calculate=pixels_to_calculate)
    # total_list = deconstruct_matrix(dataset_2d[pixels_to_calculate,:].transpose(), C_b.transpose())
    # A_mask[pixels_to_calculate] = total_list.transpose()
    #
    # A_b = A_mask
    # A_b = A_b.reshape([eigen_norm_image.shape[0], eigen_norm_image.shape[1]])
    #
    # save_image(A_b, "A_background.png")
    # print("A_background",time.time()-time_all)
    # save_image(np.mean(dataset_2d - A_b.reshape([-1, 1]) * C_b, axis=1).reshape([eigen_norm_image.shape[0], eigen_norm_image.shape[1]]), "A_wo_background_mean.png")
    #
    # C = deconstruct_matrix(dataset_2d - A_b.reshape([-1, 1]) * C_b,reshape_to_2d_over_time(A))
    # dask.compute([dask.delayed(optimize.nnls)(
    # dataset_2d - A_b.reshape([-1, 1]) * C_b, reshape_to_2d_over_time(A)[:, x]) for x
    #               in range(A.shape[0])])
    # C = np.vstack([x[0] for x in C[0]]).transpose()
    # print("All", time.time()-time_all)
    return C, C_b


def iterative_solve(dataset_2d, A, C_b, pixels_to_calculate):
    dataset = dataset_2d[pixels_to_calculate, :]
    A = reshape_to_2d_over_time(A)[pixels_to_calculate, :]

    # A_non_zero = A[A!=0]
    # A = A*4/np.percentile(A_non_zero,2)
    A_non_zero = A[A != 0]

    print("Apercent", np.percentile(A_non_zero, 5), np.percentile(A_non_zero, 15),
          np.percentile(A_non_zero, 25), np.percentile(A_non_zero, 35),
          np.percentile(A_non_zero, 45), np.percentile(A_non_zero, 45))
    C = np.zeros((A.shape[1], dataset.shape[1]))
    for x in range(A.shape[1]):
        selected_A = A[:, x]
        C[x] = np.mean(dataset[selected_A != 0], axis=0).reshape((-1, 1)).transpose()
    C = filter_close_0(C)
    # A_b = filter_close_0(np.random.normal(size=[1,len(pixels_to_calculate)],loc=0.2, scale=.1))
    A_b = filter_close_0(np.mean(dataset - np.matmul(A, C), axis=1).reshape([1, -1])).transpose()

    print(np.mean(C), np.mean(A_b))
    print("A_b", np.count_nonzero(A_b))
    print("C", np.count_nonzero(C))
    A_all = np.hstack([A, A_b])
    C_all = np.vstack([C, C_b])
    for x in range(100):
        numerator_c = np.matmul(A_all.transpose(),
                                dataset)
        denomeinator_part_1_c = np.matmul(A_all.transpose(),
                                          A_all)  # write out in comments what each matmul dimensions should be
        denomeinator_c = np.matmul(denomeinator_part_1_c, C_all)
        C_all = np.divide(np.multiply(C_all, numerator_c), denomeinator_c)
        C_all[-1:] = C_b
        print("C", C.size - np.count_nonzero(C))
        numerator_ab = np.matmul(dataset, C_all.transpose())
        denomeinator_part_1_ab = np.matmul(C_all, C_all.transpose())
        denomeinator_ab = np.matmul(A_all, denomeinator_part_1_ab)
        A_all = np.divide(np.multiply(A_all, numerator_ab), denomeinator_ab)
        A_all[:, 0:A.shape[1]] = A
        # print("A_b", A_b.size-np.count_nonzero(A_b))
    # print(np.mean(C), np.mean(A_b))
    return C_all[0:C.shape[0]], A_all[:, -1:]


def filter_close_0(data):
    data[data < 0.00000000001] = 0.00000000001
    return data


def filter_0(data):
    data[data < 0.00] = 0.00
    return data


def deconstruct_matrix(V, W, dif=1e-15):
    # V[V<0.00000000]=0.00000000
    H = np.zeros((W.shape[1], V.shape[1]))
    for x in range(W.shape[1]):
        selected_W = W[:, x]
        H[x] = np.mean(
            np.divide(V[selected_W != 0], W[:, x][selected_W != 0].reshape((-1, 1))),
            axis=0).reshape((-1, 1)).transpose()
    # cur_w = np.ones_like(W)

    top = np.matmul(W.transpose(), V)
    bottom_part_1 = np.matmul(W.transpose(), W)
    for x in range(5):
        bottom = np.matmul(bottom_part_1, H)
        H_next = np.divide(np.multiply(H, numerator), bottom)
        if np.max(np.abs(H_next - H)) <= dif:
            print(x)
            break
        H = H_next
    print(3000)
    return H_next


def time_thing(func):
    def time_inner(*args):
        time_start = time.time()
        temp = func(*args)
        print(time_start - time.time())
        return temp

    return time_inner
