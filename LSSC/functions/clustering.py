import numpy as np
from skimage import measure
from LSSC.functions.embeddings import embed_eigen

def cluster_image(e_vectors, num_clusters, original_shape, refinement=False,
                 num_eigen_vector_select=5, max_iter = 1000):
    """
    Computes the Local Selective Spectral Clustering algorithm on an set of
    eigen vectors
    Parameters
    ----------
    e_vectors eigen vector in 2D numpy array
    num_clusters
    original_shape original shape of image
    refinement if to do pixel refinement
    num_eigen_vector_select number of eigen values to project into
    max_iter max amount of pixels to try and cluster around

    Returns
    -------
    2D list of clusters [[list of pixels cluster 1], [
                          list of pixels cluster 2] ... ]
    It will have length num_clusters unless max_iter amount is surpassed
    """
    pixel_length = e_vectors.shape[0]
    e_vectors_squared = np.power(e_vectors, 2)
    pixel_embedings = np.sum(e_vectors_squared, axis=1)

    pixel_sort_indices = np.flip(np.argsort(pixel_embedings))
    cluster_list = []

    iter_counter = 0
    while len(cluster_list) < num_clusters and len(
            pixel_sort_indices) > 0 and iter_counter < max_iter:
        iter_counter += 1
        print(iter_counter, len(cluster_list))
        current_pixel_number = pixel_sort_indices[0]
        small_eigen_vectors = select_eigen_vectors(e_vectors, [current_pixel_number], num_eigen_vector_select)

        small_pixel_embedings = embed_eigen(small_eigen_vectors)
        small_pixel_distance = pixel_distance(small_eigen_vectors, current_pixel_number)
        pixels_in_cluster = np.nonzero(
            small_pixel_distance <= small_pixel_embedings)


        pixels_in_cluster_final = connected_component(pixel_length, original_shape, pixels_in_cluster, current_pixel_number)

        cluster_size_threshold = 10  # TODO also add this before and after refinement
        if len(pixels_in_cluster_final[0]) > 50:
            cluster_list.append(pixels_in_cluster_final)
            pixel_sort_indices = np.extract(
                np.in1d(pixel_sort_indices, pixels_in_cluster_final[0],
                        assume_unique=True, invert=True),
                pixel_sort_indices)  # this is correct points in clusters can't initalize a cluster
        else:
            pixel_sort_indices = np.delete(
                np.append(pixel_sort_indices, pixel_sort_indices[0]), 0)
    return cluster_list
def pixel_distance(eigen_vectors, pixel_num):
    return np.sum(np.power(eigen_vectors - eigen_vectors[pixel_num], 2),
           axis=1)

def connected_component(pixel_length, original_shape, pixels_in_cluster, current_pixel_number):
    original_zeros = np.zeros((pixel_length))
    original_zeros[pixels_in_cluster] = 1
    pixel_image = np.reshape(original_zeros, original_shape[1:])
    blobs_labels = np.reshape(measure.label(pixel_image, background=0),
                              (-1))
    correct_label = blobs_labels[current_pixel_number]
    pixels_in_cluster_new = np.nonzero(
        blobs_labels == correct_label)
    return pixels_in_cluster_new
def select_eigen_vectors(e_vectors, pixels, num_eigen_vector_select):
    pixel_eigen_vec_values = np.sum(e_vectors[pixels],axis=0)
    pixel_eigen_vec_values_sort_indices = np.flip(
        np.argsort(pixel_eigen_vec_values))
    small_eigen_vectors = e_vectors[:,
                          pixel_eigen_vec_values_sort_indices[
                          :num_eigen_vector_select]]
    return small_eigen_vectors