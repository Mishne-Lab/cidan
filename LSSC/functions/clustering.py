import numpy as np
from skimage import measure
from LSSC.functions.embeddings import embed_eigen_norm
from typing import Union, Any, List, Optional, cast, Tuple, Dict
from functools import reduce
from itertools import compress
from scipy.ndimage.morphology import binary_fill_holes
from LSSC.Parameters import Parameters


def cluster_image(e_vectors: np.ndarray,
                  original_shape: tuple, original_2d_vol: np.ndarray,
                  parameters: Parameters) -> List[np.ndarray]:
    """
    Computes the Local Selective Spectral Clustering algorithm on an set of
    eigen vectors
    Parameters
    ----------
    e_vectors: Eigen vector in 2D numpy array
    original_shape: Original shape of image
    original_2d_vol: A flattened 2d volume of the original image, used for
        mergestep
    parameters: A parameter object, used params:
        num_clusters: Number of clusters
        refinement: If to do cluster refinement
        num_eigen_vector_select: Number of eigen values to project into
        max_iter: Max amount of pixels to try and cluster around
        cluster_size_threshold: Min size for cluster to be output if too big might
         limit number of clusters
        fill_holes: fills holes in clusters
        elbow: whether to use elbow thresholding in refinement step
        eigen_threshold_method: whether to use thresholding method when
            selecting eigen values
        eigen_threshold_value: value for said method
    Returns
    -------
    2D list of clusters [[np.array of pixels cluster 1], [
                          np.array  of pixels cluster 2] ... ]
    It will have length num_clusters unless max_iter amount is surpassed
    """
    pixel_length = e_vectors.shape[0]
    pixel_embedings = embed_eigen_norm(
        e_vectors)  # embeds the pixels in the eigen space
    initial_pixel_list = np.flip(np.argsort(
        pixel_embedings))  # creates a list of pixels with the highest values
    # in the eigenspace this list is used to decide on the initial point
    # for the cluster

    cluster_list = []  # output list of clusters

    # iter_counter is used to limit the ammount of pixels it tries
    # from initial_pixel_list
    iter_counter = 0
    while len(cluster_list) < parameters.num_clusters and len(
            initial_pixel_list) > 0 and iter_counter < parameters.max_iter:
        iter_counter += 1
        print(iter_counter, len(cluster_list),
              len(cluster_list[-1]) if len(cluster_list) > 0 else 0)
        initial_pixel = initial_pixel_list[0]  # Select initial point
        # select eigen vectors to project into
        small_eigen_vectors = select_eigen_vectors(e_vectors,
                                                   [initial_pixel],
                                                   parameters.num_eigen_vector_select,
                                                   threshold_method=parameters.eigen_threshold_method,
                                                   threshold=parameters.eigen_threshold_value)
        # TODO Find way to auto determin threshold value automatically max values
        # project into new eigen space
        small_pixel_embeding_norm = embed_eigen_norm(small_eigen_vectors)

        # calculate the distance between the initial point and each pixel
        # in the new eigen space
        small_pixel_distance = pixel_distance(small_eigen_vectors,
                                              initial_pixel)

        # selects pixels in cluster
        pixels_in_cluster = np.nonzero(
            small_pixel_distance <= small_pixel_embeding_norm)[0]

        # runs a connected component analysis around the initial point
        # in original image
        pixels_in_cluster_comp = connected_component(pixel_length,
                                                     original_shape,
                                                     pixels_in_cluster,
                                                     initial_pixel)
        pixels_in_cluster_final = pixels_in_cluster_comp

        # runs refinement step if enabled and if enough pixels in cluster
        if parameters.refinement:  # TODO Look at this again
            # and len(pixels_in_cluster_final) > \
            #  cluster_size_threshold / 2
            # selects a new set of eigenvectors based on the pixels in cluster
            rf_eigen_vectors = select_eigen_vectors(e_vectors,
                                                    pixels_in_cluster_final,
                                                    parameters.num_eigen_vector_select,
                                                    threshold_method=parameters.eigen_threshold_method,
                                                    threshold=parameters.eigen_threshold_value)

            # embeds all pixels in this new eigen space
            rf_pixel_embedding_norm = embed_eigen_norm(rf_eigen_vectors)

            # selects the initial point based on the pixel with max in
            # the new embedding space
            rf_initial_point = rf_select_initial_point(rf_pixel_embedding_norm,
                                                       pixels_in_cluster_final)

            # calculate the distance between the initial point and each pixel
            # in the new eigen space
            rf_pixel_distance = pixel_distance(rf_eigen_vectors,
                                               rf_initial_point)

            # selects pixels in cluster
            if parameters.elbow_threshold_method:
                threshold = elbow_threshold(rf_pixel_distance,
                                            np.argsort(rf_pixel_distance))
                rf_pixels_in_cluster = np.nonzero(
                    rf_pixel_distance < threshold)[0]
            else:
                rf_pixels_in_cluster = np.nonzero(
                    rf_pixel_distance <= rf_pixel_embedding_norm)[0]

            # runs a connected component analysis around the initial point
            # in original image
            rf_pixels_in_cluster_comp = connected_component(pixel_length,
                                                            original_shape,
                                                            rf_pixels_in_cluster,
                                                            rf_initial_point)
            pixels_in_cluster_final = rf_pixels_in_cluster_comp

        # checks if cluster is big enough
        print(len(pixels_in_cluster_final))

        if parameters.cluster_size_threshold < len(
                pixels_in_cluster_final) < parameters.cluster_size_limit:
            cluster_list.append(pixels_in_cluster_final)

            # takes all pixels in current cluster out of initial_pixel_list
            initial_pixel_list = np.extract(
                np.in1d(initial_pixel_list, pixels_in_cluster_final,
                        assume_unique=True, invert=True),
                initial_pixel_list)
            if initial_pixel not in pixels_in_cluster_final:
                initial_pixel_list = np.delete(initial_pixel_list, 0)

            print(len(initial_pixel_list))
        else:
            # takes current initial point and moves it to end of
            # initial_pixel_list
            initial_pixel_list = np.delete(
                np.append(initial_pixel_list, initial_pixel_list[0]), 0)
    if parameters.fill_holes:
        cluster_list = fill_holes(cluster_list, pixel_length, original_shape)
    # Merges clusters
    cluster_list = merge_clusters(cluster_list,
                                  temporal_coefficient=parameters.merge_temporal_coef,
                                  original_2d_vol=original_2d_vol)
    if parameters.fill_holes:
        cluster_list = fill_holes(cluster_list, pixel_length, original_shape)
    return cluster_list


def fill_holes(cluster_list: List[np.ndarray], pixel_length: int,
               original_shape: Tuple[int]) -> List[np.ndarray]:
    """
    Close holes in each cluster
    Parameters
    ----------
    cluster_list
    pixel_length
    original_shape

    Returns
    -------
    cluster list with holes filled
    """
    for num, cluster in enumerate(cluster_list):
        original_zeros = np.zeros((pixel_length))
        original_zeros[cluster] = 255
        image_filled = binary_fill_holes(np.reshape(original_zeros,
                                                    original_shape[1:]))
        image_filled_2d = np.reshape(image_filled, (-1))
        cluster_list[num] = np.nonzero(image_filled_2d)[0]
    return cluster_list


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
                        pixels_in_cluster: np.ndarray,
                        initial_pixel_number: int) -> np.ndarray:
    """
    Runs a connected component analysis on a group of pixels in an image
    Parameters
    ----------
    pixel_length: Number of pixels in image
    original_shape: Ariginal shape of image
    pixels_in_cluster: A tuple with the first entry as a
    initial_pixel_number: The number of the original pixel in the
        flattened image


    Returns
    -------
    An subset of the original pixels that are connected to initial pixel
    """
    # TODO add in im fill before connected component
    # first creates an image with pixel values of 1 if pixel in cluster
    original_zeros = np.zeros(pixel_length)
    original_zeros[pixels_in_cluster] = 1
    pixel_image = np.reshape(original_zeros, original_shape[1:])

    # runs connected component analysis on image
    blobs_labels = np.reshape(measure.label(pixel_image, background=0),
                              (-1))
    correct_label = blobs_labels[initial_pixel_number]

    # filters pixels to only ones with same label as initial pixel
    pixels_in_cluster_new = np.nonzero(
        blobs_labels == correct_label)[0]
    return pixels_in_cluster_new


def select_eigen_vectors(eigen_vectors: np.ndarray,
                         pixels_in_cluster: np.ndarray,
                         num_eigen_vector_select: int,
                         threshold_method: bool = False,
                         threshold: float = .1) -> np.ndarray:
    """
    Selects eigen vectors that are most descriptive of a set a points
    Parameters
    ----------
    eigen_vectors: The eigen vectors describing the vector space with
        dimensions number of pixels in image by number of eigen vectors
    pixels_in_cluster: Np array of indices of all pixels in cluster
    num_eigen_vector_select: Number of eigen vectors to select
    threshold_method: this is a bool on whether to run the threshold method to select the eigen vectors
    threshold

    Returns
    -------
    the eigen vectors describing the new vector space with
        dimensions number of pixels in image by numb_eigen_vector_select

    """
    pixel_eigen_vec_values = np.abs(np.sum(eigen_vectors[pixels_in_cluster],
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
                            pixels_in_cluster: np.ndarray):
    """
    Selects an initial point for clustering based on the pixels in current
    cluster, this is part of the refinement step
    Parameters
    ----------
    pixel_embedings: The embedings of each pixel in a vector space
    pixels_in_cluster: a list of the indices of the pixel in the cluster

    Returns
    -------
    an indice for the initial point pixel
    """
    indice_in_cluster = \
        np.flip(np.argsort(pixel_embedings[pixels_in_cluster]))[0]
    return np.sort(pixels_in_cluster)[indice_in_cluster]


def elbow_threshold(pixel_vals, pixel_val_sort_indices, half=True):
    n_points = len(pixel_vals) if not half else len(pixel_vals) // 2
    pixel_vals_sorted_zipped = np.array(list(
        zip(range(n_points), pixel_vals[pixel_val_sort_indices[:n_points]])))
    # xnew = np.linspace(0, n_points, 300)
    # spl = make_interp_spline(list(range(n_points)), pixel_vals[pixel_val_sort_indices[:n_points]], k=101)  # BSpline object
    # power_smooth = spl(xnew)
    #
    # plt.scatter(list(range(n_points)),pixel_vals[pixel_val_sort_indices[:n_points]])
    # plt.savefig("/data2/Sam/pythonTestEnviroment/output_images/plots/dist_plot_1.png")
    first_point = pixel_vals_sorted_zipped[0, :]
    last_point = pixel_vals_sorted_zipped[-1, :]
    line_vec = last_point - first_point
    line_vec_norm = line_vec / (np.sum(np.power(line_vec, 2)) ** .5)
    dist_from_first = pixel_vals_sorted_zipped - first_point
    scalar_product = dist_from_first * np.transpose(line_vec_norm)
    vec_to_line = dist_from_first - scalar_product * line_vec_norm
    dist_to_line = np.power(np.sum(np.power(vec_to_line, 2), axis=1), .5)
    dist_max_indice = np.argmax(dist_to_line)
    threshold = pixel_vals_sorted_zipped[dist_max_indice][1]
    return threshold


def merge_clusters(cluster_list: List,
                   temporal_coefficient: float, original_2d_vol: np.ndarray):
    # TODO is this the most efficient implementation I can do
    """
    Merges clusters based on temporal and spacial overlap
    Parameters
    ----------
    eigen_vectors
    cluster_list
    temporal_coefficient

    Returns
    -------

    """
    new_clusters = []
    while True:
        combined_clusters = False
        while len(cluster_list) > 0:
            curr_cluster = cluster_list.pop(0)
            similar_clusters_bool = list(map(lambda comp_cluster:
                                             compare_cluster(curr_cluster,
                                                             comp_cluster,
                                                             temporal_coefficient,
                                                             original_2d_vol),
                                             cluster_list))
            similar_clusters = list(
                compress(cluster_list, similar_clusters_bool))
            cluster_list = list(compress(cluster_list, map(lambda x: not x,
                                                           similar_clusters_bool)))
            combined_clusters = True if len(
                similar_clusters) > 0 else combined_clusters
            curr_cluster_combined = list(reduce(
                lambda cluster1, cluster2: combine_clusters(cluster1,
                                                            cluster2),
                [curr_cluster] + similar_clusters))
            new_clusters.append(curr_cluster)
        if not combined_clusters:
            break
        else:
            cluster_list = new_clusters
            new_clusters = []

    return new_clusters


def compare_cluster(cluster1: List[int],
                    cluster2: List[int], temporal_coefficient: int,
                    original_2d_vol: np.ndarray) -> bool:
    """
    Compares two clusters and sees if they meet the standards for being combined #calculate average time trace given spacial support correlation is above a certain factor
    Parameters
    ----------
    cluster1
    cluster2
    temporal_coefficient
    original_2d_vol

    Returns
    -------
    """

    if any([x in cluster2 for x in cluster1]):  # check spacial simlarity
        cluster1_time_trace_avg = np.mean(original_2d_vol[cluster1],
                                          axis=0)  # TODO Do I zscore the time trace
        cluster2_time_trace_avg = np.mean(original_2d_vol[cluster2], axis=0)
        cluster1_time_trace_avg_z_scored = (cluster1_time_trace_avg - np.mean(
            cluster1_time_trace_avg)) / np.std(cluster1_time_trace_avg)
        cluster2_time_trace_avg_z_scored = (cluster2_time_trace_avg - np.mean(
            cluster2_time_trace_avg)) / np.std(cluster2_time_trace_avg)
        if np.linalg.norm(
                cluster1_time_trace_avg_z_scored -
                cluster2_time_trace_avg_z_scored) < temporal_coefficient * (
                original_2d_vol.shape[0] ** .5):
            return True

    return False


def combine_clusters(cluster1: List[int], cluster2: List[int]) -> List[int]:
    """

    Parameters
    ----------
    cluster1
    cluster2

    Returns
    -------

    """
    cluster2_not_in_1 = list(filter(lambda x: x not in cluster1, cluster2))
    return np.array(list(cluster1) + cluster2_not_in_1)
