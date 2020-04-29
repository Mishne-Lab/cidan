import numpy as np
from skimage import measure
from CIDAN.LSSC.functions.embeddings import embed_eigen_norm
from typing import Union, Any, List, Optional, cast, Tuple, Dict
from functools import reduce
from itertools import compress
from scipy.ndimage.morphology import binary_fill_holes
from matplotlib import pyplot as plt
from scipy.sparse.csgraph import connected_components as connected_components_graph
from scipy.sparse import csr_matrix
from dask import delayed
@delayed
def roi_extract_image(*, e_vectors: np.ndarray,
                      original_shape: tuple, original_2d_vol: np.ndarray, merge: bool,
                      num_rois: int, refinement: bool, num_eigen_vector_select: int,
                      max_iter: int, roi_size_min: int, fill_holes: bool,
                      elbow_threshold_method: bool, elbow_threshold_value: float,
                      eigen_threshold_method: bool,
                      eigen_threshold_value: float, merge_temporal_coef: float,
                      roi_size_limit: int, box_num:int) -> List[np.ndarray]:
    """
    Computes the Local Selective Spectral roi_extraction algorithm on an set of
    eigen vectors
    Parameters
    ----------
    e_vectors: Eigen vector in 2D numpy array
    original_shape: Original shape of image
    original_2d_vol: A flattened 2d volume of the original image, used for
        mergestep
    parameters: A parameter object, used params:
        num_rois: Number of rois
        refinement: If to do roi refinement
        num_eigen_vector_select: Number of eigen values to project into
        max_iter: Max amount of pixels to try and roi around
        roi_size_threshold: Min size for roi to be output if too big might
         limit number of rois
        fill_holes: fills holes in rois
        elbow: whether to use elbow thresholding in refinement step
        eigen_threshold_method: whether to use thresholding method when
            selecting eigen values
        eigen_threshold_value: value for said method
    Returns
    -------
    2D list of rois [[np.array of pixels roi 1], [
                          np.array  of pixels roi 2] ... ]
    It will have length num_rois unless max_iter amount is surpassed
    """
    print("Spatial Box {}: Starting ROI selection process".format(box_num))
    pixel_length = e_vectors.shape[0]
    pixel_embedings = embed_eigen_norm(
        e_vectors)  # embeds the pixels in the eigen space
    initial_pixel_list = np.flip(np.argsort(
        pixel_embedings))  # creates a list of pixels with the highest values
    # in the eigenspace this list is used to decide on the initial point
    # for the roi

    roi_list = []  # output list of rois

    # iter_counter is used to limit the ammount of pixels it tries
    # from initial_pixel_list
    iter_counter = 0
    while len(roi_list) < num_rois and len(
            initial_pixel_list) > 0 and iter_counter < max_iter:
        iter_counter += 1
        # print(iter_counter, len(roi_list),
        #       len(roi_list[-1]) if len(roi_list) > 0 else 0)
        initial_pixel = initial_pixel_list[0]  # Select initial point
        # select eigen vectors to project into
        small_eigen_vectors = select_eigen_vectors(e_vectors,
                                                   [initial_pixel],
                                                   num_eigen_vector_select,
                                                   threshold_method=eigen_threshold_method,
                                                   threshold=eigen_threshold_value)
        # print("original",small_eigen_vectors.shape)
        # TODO Find way to auto determin threshold value automatically max values
        # project into new eigen space
        small_pixel_embeding_norm = embed_eigen_norm(small_eigen_vectors)

        # calculate the distance between the initial point and each pixel
        # in the new eigen space
        small_pixel_distance = pixel_distance(small_eigen_vectors,
                                              initial_pixel)

        # selects pixels in roi
        pixels_in_roi = np.nonzero(
            small_pixel_distance <= small_pixel_embeding_norm)[0]

        # runs a connected component analysis around the initial point
        # in original image
        pixels_in_roi_comp = connected_component(pixel_length,
                                                     original_shape,
                                                     pixels_in_roi,
                                                     initial_pixel)
        pixels_in_roi_final = pixels_in_roi_comp

        # runs refinement step if enabled and if enough pixels in roi
        if refinement:  # TODO Look at this again
            # and len(pixels_in_roi_final) > \
            #  roi_size_threshold / 2
            # selects a new set of eigenvectors based on the pixels in roi
            rf_eigen_vectors = select_eigen_vectors(e_vectors,
                                                    pixels_in_roi_final,
                                                    num_eigen_vector_select,
                                                    threshold_method=eigen_threshold_method,
                                                    threshold=eigen_threshold_value)

            # print("rf",rf_eigen_vectors.shape)
            # embeds all pixels in this new eigen space
            rf_pixel_embedding_norm = embed_eigen_norm(rf_eigen_vectors)

            # selects the initial point based on the pixel with max in
            # the new embedding space
            rf_initial_point = rf_select_initial_point(rf_pixel_embedding_norm,
                                                       pixels_in_roi_final)

            # calculate the distance between the initial point and each pixel
            # in the new eigen space
            rf_pixel_distance = pixel_distance(rf_eigen_vectors,
                                               rf_initial_point)

            # selects pixels in roi
            if elbow_threshold_method:
                threshold = elbow_threshold_value * elbow_threshold(rf_pixel_distance,
                                                                     np.argsort(
                                                                         rf_pixel_distance),
                                                                     half=True,
                                                                     num=iter_counter)
                rf_pixels_in_roi = np.nonzero(
                    rf_pixel_distance < threshold)[0]
            else:
                rf_pixels_in_roi = np.nonzero(
                    rf_pixel_distance <= rf_pixel_embedding_norm)[0]

            # runs a connected component analysis around the initial point
            # in original image
            rf_pixels_in_roi_comp = connected_component(pixel_length,
                                                            original_shape,
                                                            rf_pixels_in_roi,
                                                            rf_initial_point)
            pixels_in_roi_final = rf_pixels_in_roi_comp

        # checks if roi is big enough
        # print("roi size:", len(pixels_in_roi_final))

        if roi_size_min < len(
                pixels_in_roi_final) < roi_size_limit:
            roi_list.append(pixels_in_roi_final)

            # takes all pixels in current roi out of initial_pixel_list
            initial_pixel_list = np.extract(
                np.in1d(initial_pixel_list, pixels_in_roi_final,
                        assume_unique=True, invert=True),
                initial_pixel_list)
            if initial_pixel not in pixels_in_roi_final:
                initial_pixel_list = np.delete(initial_pixel_list, 0)

            # print(len(initial_pixel_list))
        else:
            # takes current initial point and moves it to end of
            # initial_pixel_list
            initial_pixel_list = np.delete(
                np.append(initial_pixel_list, initial_pixel_list[0]), 0)
    if fill_holes:
        roi_list = fill_holes_func(roi_list, pixel_length, original_shape)
    # Merges rois
    if merge:
        roi_list = merge_rois(roi_list,
                                  temporal_coefficient=merge_temporal_coef,
                                  original_2d_vol=original_2d_vol)
        if fill_holes:
            roi_list = fill_holes_func(roi_list, pixel_length, original_shape)
    return roi_list


def fill_holes_func(roi_list: List[np.ndarray], pixel_length: int,
               original_shape: Tuple[int]) -> List[np.ndarray]:
    """
    Close holes in each roi
    Parameters
    ----------
    roi_list
    pixel_length
    original_shape

    Returns
    -------
    roi list with holes filled
    """
    for num, roi in enumerate(roi_list):
        original_zeros = np.zeros((pixel_length))
        original_zeros[roi] = 255
        image_filled = binary_fill_holes(np.reshape(original_zeros,
                                                    original_shape[1:]))
        image_filled_2d = np.reshape(image_filled, (-1))
        roi_list[num] = np.nonzero(image_filled_2d)[0]
    return roi_list


def pixel_distance(eigen_vectors: np.ndarray, pixel_num: int) -> np.ndarray:
    """
    Calculates squared distance between pixels in embedding space and initial_point
    Parameters
    ----------
    eigen_vectors: The eigen vectors describing the vector space with
        dimensions number of pixels in image by number of eigen vectors
    pixel_num: The number of the initial pixel in the eigen vectors

    Returns
    -------
    A np array with dim: number of pixels in image
    """
    return np.sum(np.power(eigen_vectors - eigen_vectors[pixel_num], 2),
                  axis=1)


def connected_component(pixel_length: int, original_shape: Tuple[int],
                        pixels_in_roi: np.ndarray,
                        initial_pixel_number: int) -> np.ndarray:
    """
    Runs a connected component analysis on a group of pixels in an image
    Parameters
    ----------
    pixel_length: Number of pixels in image
    original_shape: Ariginal shape of image
    pixels_in_roi: A tuple with the first entry as a
    initial_pixel_number: The number of the original pixel in the
        flattened image


    Returns
    -------
    An subset of the original pixels that are connected to initial pixel
    """
    # TODO add in im fill before connected component
    # first creates an image with pixel values of 1 if pixel in roi
    original_zeros = np.zeros(pixel_length)
    original_zeros[pixels_in_roi] = 1
    pixel_image = np.reshape(original_zeros, original_shape[1:])

    # runs connected component analysis on image
    blobs_labels = np.reshape(measure.label(pixel_image, background=0),
                              (-1))
    correct_label = blobs_labels[initial_pixel_number]

    # filters pixels to only ones with same label as initial pixel
    pixels_in_roi_new = np.nonzero(
        blobs_labels == correct_label)[0]
    return pixels_in_roi_new


def select_eigen_vectors(eigen_vectors: np.ndarray,
                         pixels_in_roi: np.ndarray,
                         num_eigen_vector_select: int,
                         threshold_method: bool = False,
                         threshold: float = .1) -> np.ndarray:
    """
    Selects eigen vectors that are most descriptive of a set a points
    Parameters
    ----------
    eigen_vectors: The eigen vectors describing the vector space with
        dimensions number of pixels in image by number of eigen vectors
    pixels_in_roi: Np array of indices of all pixels in roi
    num_eigen_vector_select: Number of eigen vectors to select
    threshold_method: this is a bool on whether to run the threshold method to select the eigen vectors
    threshold

    Returns
    -------
    the eigen vectors describing the new vector space with
        dimensions number of pixels in image by numb_eigen_vector_select

    """
    pixel_eigen_vec_values = np.abs(np.sum(eigen_vectors[pixels_in_roi],
                                           axis=0))  # TODO DONE add in absolute value for only the non-refinement step and add absolute value of the sum
    pixel_eigen_vec_values_sort_indices = np.flip(
        np.argsort(pixel_eigen_vec_values))
    if threshold_method:

        threshold_filter = pixel_eigen_vec_values > threshold * \
                           pixel_eigen_vec_values[
                               pixel_eigen_vec_values_sort_indices[0]]
        small_eigen_vectors = eigen_vectors[:, np.nonzero(threshold_filter)[0]]

    else:
        pixel_eigen_vec_values_sort_indices = np.flip(
            np.argsort(
                pixel_eigen_vec_values))  # TODO DONE add thresholding method to this too .1 is a good threshold
        small_eigen_vectors = eigen_vectors[:,
                              pixel_eigen_vec_values_sort_indices[
                              :num_eigen_vector_select]]
    return small_eigen_vectors


def rf_select_initial_point(pixel_embedings: np.ndarray,
                            pixels_in_roi: np.ndarray):
    """
    Selects an initial point for roi_extraction based on the pixels in current
    roi, this is part of the refinement step
    Parameters
    ----------
    pixel_embedings: The embedings of each pixel in a vector space
    pixels_in_roi: a list of the indices of the pixel in the roi

    Returns
    -------
    an indice for the initial point pixel
    """
    indice_in_roi = \
        np.flip(np.argsort(pixel_embedings[pixels_in_roi]))[0]
    return np.sort(pixels_in_roi)[indice_in_roi]


def elbow_threshold(pixel_vals: np.ndarray, pixel_val_sort_indices: np.ndarray,
                    half: bool = True, num=0) -> float:
    """
    Calculates the elbow threshold for the refinement step in roi_extraction
    algorithm, determines the pixel that is farthest away from line drawn
    from first to last or first to middle pixel

    Projects each point(pixel # in sorted pixel_val, distance val) to the line then subtracts that from the points value to find distance
    Parameters
    ----------
    pixel_vals: The distance values for each pixel
    pixel_val_sort_indices: The array necessary to sort said pixels from lowest
     to highest
    half: whether to run only with the closest half of the pixels, recommended

    Returns
    -------
    float, the optimal threshold based on elbow
    """
    n_points = len(pixel_vals) if not half else len(pixel_vals) // 2
    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.scatter(list(range(n_points)),
    #            pixel_vals[pixel_val_sort_indices[:n_points]])
    # fig.savefig(
    #     "/data2/Sam/pythonTestEnviroment/output_images/plots/dist_plot_{}1.png".format(num),
    #     aspect='auto')
    # plt.close('all')
    pixel_vals_sorted_zipped = np.array(list(
        zip(range(n_points), pixel_vals[pixel_val_sort_indices[:n_points]])))

    first_point = pixel_vals_sorted_zipped[0, :]
    last_point = pixel_vals_sorted_zipped[-1, :]
    line_vec = last_point - first_point

    line_vec_norm = line_vec / (np.sum(np.power(line_vec, 2)) ** .5)
    dist_from_first = pixel_vals_sorted_zipped - first_point
    proj_point_to_line = np.matmul(dist_from_first,
                                   line_vec_norm[:, None]) * line_vec_norm
    vec_to_line = dist_from_first - proj_point_to_line
    dist_to_line = np.power(np.sum(np.power(vec_to_line, 2), axis=1), .5)
    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.scatter(list(range(n_points)),
    #            dist_from_first[:n_points,1])
    # fig.savefig(
    #     "/Users/sschickler/Documents/LSSC-python/output_images/plots/dist_plot_{}2.png".format(num),
    #     aspect='auto')
    dist_max_indice = np.argmax(dist_to_line)
    threshold = pixel_vals_sorted_zipped[dist_max_indice][1]
    return threshold


def merge_rois(roi_list: List,
               temporal_coefficient: float, original_2d_vol: np.ndarray):
    # TODO is this the most efficient implementation I can do
    """
    Merges rois based on temporal and spacial overlap
    Parameters
    ----------
    eigen_vectors
    roi_list
    temporal_coefficient

    Returns
    -------

    """
    A = np.zeros([original_2d_vol.shape[0], len(roi_list)], dtype=bool)
    for num, cluster in enumerate(roi_list):
        A[cluster,num] = True
    A_graph = np.matmul(A.transpose(),A)
    A_csr = csr_matrix(A_graph)
    connected = connected_components_graph(A_csr,False,return_labels=True)
    roi_groups = [[] for _ in range(len(roi_list))]
    for num in range(len(roi_list)):
        roi_groups[connected[1][num]].append(roi_list[num])

    new_rois = []
    for group in roi_groups:
        group_zipped = list(enumerate(group))
        timetraces = [np.mean(original_2d_vol[roi], axis=0) for roi in group]
        while len(group_zipped)>0:
            first_num, first_roi = group_zipped.pop(0)
            rois_to_merge = []
            for num, roi in enumerate(group_zipped):
                if compare_time_traces(timetraces[first_num],timetraces[roi[0]])>temporal_coefficient:
                    rois_to_merge.append(num)
            first_roi = list(reduce(combine_rois, [first_roi]+[group_zipped[x][1] for x in rois_to_merge]))
            for num in rois_to_merge[::-1]:
                group_zipped.pop(num)
            new_rois.append(np.array(first_roi))

    return new_rois
def compare_time_traces(trace_1, trace_2):
    """
    Compares two timetraces based on person correlation
    Parameters
    ----------
    trace_1
    trace_2

    Returns
    -------
    the correlation
    """
    trace_1_mean = np.mean(trace_1)
    trace_2_mean = np.mean(trace_2)
    trace_1_sub_mean = (trace_1-trace_1_mean)
    trace_2_sub_mean = (trace_2-trace_2_mean)
    top = np.dot(trace_1_sub_mean, trace_2_sub_mean)
    bottom = (np.dot(trace_1_sub_mean, trace_1_sub_mean)*
              np.dot(trace_2_sub_mean,trace_2_sub_mean))**.5
    return top/bottom

def compare_roi(roi1: List[int],
                    roi2: List[int], temporal_coefficient: int,
                    original_2d_vol: np.ndarray) -> bool:
    """
    Compares two rois and sees if they meet the standards for being combined #calculate average time trace given spacial support correlation is above a certain factor
    Parameters
    ----------
    roi1
    roi2
    temporal_coefficient
    original_2d_vol

    Returns
    -------
    """

    if any([x in roi2 for x in roi1]):  # check spacial simlarity
        roi1_time_trace_avg = np.mean(original_2d_vol[roi1],
                                          axis=0)  # TODO Do I zscore the time trace
        roi2_time_trace_avg = np.mean(original_2d_vol[roi2], axis=0)
        roi1_time_trace_avg_z_scored = (roi1_time_trace_avg - np.mean(
            roi1_time_trace_avg)) / np.std(roi1_time_trace_avg)
        roi2_time_trace_avg_z_scored = (roi2_time_trace_avg - np.mean(
            roi2_time_trace_avg)) / np.std(roi2_time_trace_avg)
        if np.linalg.norm(
                roi1_time_trace_avg_z_scored -
                roi2_time_trace_avg_z_scored) < temporal_coefficient * (
                original_2d_vol.shape[0] ** .5):
            return True

    return False


def combine_rois(roi1: List[int], roi2: List[int]) -> List[int]:
    """
    Combines two lists of clusters into one
    Parameters
    ----------
    roi1 One list of pixels in cluster each pixel # is based on 1d representation of
        image
    roi2 List for other ROI

    Returns
    -------
    List of merged ROI
    """
    roi2_not_in_1 = list(filter(lambda x: x not in roi1, roi2))
    return np.array(list(roi1) + roi2_not_in_1)
