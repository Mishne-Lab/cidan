import numpy as np
from skimage import measure
from LSSC.functions.embeddings import embed_eigen
from typing import Union, Any, List, Optional, cast, Tuple, Dict


def cluster_image(e_vectors: np.ndarray, num_clusters: int,
                  original_shape: tuple, refinement: bool = False,
                  num_eigen_vector_select: int = 4, max_iter: int = 100000,
                  cluster_size_threshold: int = 30) -> List[np.ndarray]:
    """
    Computes the Local Selective Spectral Clustering algorithm on an set of
    eigen vectors
    Parameters
    ----------
    e_vectors: Eigen vector in 2D numpy array
    num_clusters: Number of clusters
    original_shape: Original shape of image
    refinement: If to do cluster refinement
    num_eigen_vector_select: Number of eigen values to project into
    max_iter: Max amount of pixels to try and cluster around
    cluster_size_threshold: Min size for cluster to be output if too big might
     limit number of clusters

    Returns
    -------
    2D list of clusters [[np.array of pixels cluster 1], [
                          np.array  of pixels cluster 2] ... ]
    It will have length num_clusters unless max_iter amount is surpassed
    """
    pixel_length = e_vectors.shape[0]
    pixel_embedings = embed_eigen(
        e_vectors)  # embeds the pixels in the eigen space
    initial_pixel_list = np.flip(np.argsort(
        pixel_embedings))  # creates a list of pixels with the highest values
    # in the eigenspace this list is used to decide on the initial point
    # for the cluster

    cluster_list = [] # output list of clusters

    # iter_counter is used to limit the ammount of pixels it tries
    # from initial_pixel_list
    iter_counter = 0
    while len(cluster_list) < num_clusters and len(
            initial_pixel_list) > 0 and iter_counter < max_iter:
        iter_counter += 1
        print(iter_counter, len(cluster_list),
              len(cluster_list[-1]) if len(cluster_list) > 0 else 0)
        initial_pixel = initial_pixel_list[0] # Select initial point
        # select eigen vectors to project into
        small_eigen_vectors = select_eigen_vectors(e_vectors,
                                                   [initial_pixel],
                                                   num_eigen_vector_select)
        # project into new eigen space
        small_pixel_embedings = embed_eigen(small_eigen_vectors)

        # calculate the distance between the initial point and each pixel
        # in the new eigen space
        small_pixel_distance = pixel_distance(small_eigen_vectors,
                                              initial_pixel)
        # selects pixels in cluster
        pixels_in_cluster = np.nonzero(
            small_pixel_distance <= small_pixel_embedings)[0]

        # runs a connected component analysis around the initial point
        # in original image
        pixels_in_cluster_comp = connected_component(pixel_length,
                                                     original_shape,
                                                     pixels_in_cluster,
                                                     initial_pixel)
        pixels_in_cluster_final = pixels_in_cluster_comp

        # runs refinement step if enabled and if enough pixels in cluster
        if refinement and len(pixels_in_cluster_final) > \
                cluster_size_threshold:

            # selects a new set of eigenvectors based on the pixels in cluster
            rf_eigen_vectors = select_eigen_vectors(e_vectors,
                                                    pixels_in_cluster_final[0],
                                                    num_eigen_vector_select)

            # embeds all pixels in this new eigen space
            rf_pixel_embedding = embed_eigen(rf_eigen_vectors)

            # selects the initial point based on the pixel with max in
            # the new embedding space
            rf_initial_point = rf_select_initial_point(rf_pixel_embedding,
                                                    pixels_in_cluster_final)

            # calculate the distance between the initial point and each pixel
            # in the new eigen space
            rf_pixel_distance = pixel_distance(rf_eigen_vectors,
                                               rf_initial_point)

            # selects pixels in cluster
            rf_pixels_in_cluster = np.nonzero(
                rf_pixel_distance <= rf_pixel_embedding)[0]

            # runs a connected component analysis around the initial point
            # in original image
            rf_pixels_in_cluster_comp = connected_component(pixel_length,
                                                            original_shape,
                                                            rf_pixels_in_cluster,
                                                            rf_initial_point)
            pixels_in_cluster_final = rf_pixels_in_cluster_comp

        # checks if cluster is big enough
        if len(pixels_in_cluster_final) > cluster_size_threshold:
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
    return cluster_list


def pixel_distance(eigen_vectors: np.ndarray, pixel_num:int) -> np.ndarray:
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


def connected_component(pixel_length: int, original_shape: Tuple[int], pixels_in_cluster: np.ndarray,
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
    # first creates an image with pixel values of 1 if pixel in cluster
    original_zeros = np.zeros(pixel_length)
    original_zeros[pixels_in_cluster] = 1
    pixel_image = np.reshape(original_zeros, original_shape[1:])

    #runs connected component analysis on image
    blobs_labels = np.reshape(measure.label(pixel_image, background=0),
                              (-1))
    correct_label = blobs_labels[initial_pixel_number]

    # filters pixels to only ones with same label as initial pixel
    pixels_in_cluster_new = np.nonzero(
        blobs_labels == correct_label)[0]
    return pixels_in_cluster_new


def select_eigen_vectors(eigen_vectors: np.ndarray, pixels_in_cluster: np.ndarray,
                         num_eigen_vector_select: int) -> np.ndarray:
    """
    Selects eigen vectors that are most descriptive of a set a points
    Parameters
    ----------
    eigen_vectors: The eigen vectors describing the vector space with
        dimensions number of pixels in image by number of eigen vectors
    pixels_in_cluster: Np array of indices of all pixels in cluster
    num_eigen_vector_select: Number of eigen vectors to select

    Returns
    -------
    the eigen vectors describing the new vector space with
        dimensions number of pixels in image by numb_eigen_vector_select

    """
    pixel_eigen_vec_values = np.sum(eigen_vectors[pixels_in_cluster], axis=0)
    pixel_eigen_vec_values_sort_indices = np.flip(
        np.argsort(pixel_eigen_vec_values))
    small_eigen_vectors = eigen_vectors[:,
                          pixel_eigen_vec_values_sort_indices[
                          :num_eigen_vector_select]]
    return small_eigen_vectors


def rf_select_initial_point(pixel_embedings: np.ndarray, pixels_in_cluster: np.ndarray):
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
