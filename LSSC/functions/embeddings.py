import os

os.environ["OMP_NUM_THREADS"] = "10"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10"  # export NUMEXPR_NUM_THREADS=6
import scipy.sparse as sparse
import hnswlib as hnsw
import numpy as np


def calc_affinity_matrix(pixel_list: np.matrix, metric="l2", knn=20,
                         accuracy=200, connections=40, num_threads=10):
    """
    Calculates an pairwise affinity matrix for the image stack
    Parameters
    ----------
    pixel_list: a 2d np array of pixels with dim 0 being a list of pixels and dim 1 being pixel values over time
    metric: can be "l2" squared l2, "ip" Inner product, "cosine" Cosine similarity
    knn: number of nearest neighbors to search for
    accuracy: time of construction vs acuracy tradeoff
    connections: max number of outgoing connections
    num_threads: number of threads to use

    Returns
    -------

    """
    assert knn < accuracy, "Knn needs to be less than the accuracy ammount"
    # TODO make connections value scale based on available memory

    dim = pixel_list.shape[1]
    num_elements = pixel_list.shape[0]

    p = hnsw.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=accuracy,
                 M=connections)
    p.add_items(pixel_list, num_threads=num_threads)
    indices, distances = p.knn_query(pixel_list, k=knn,
                                     num_threads=num_threads)  # lazy random walk means it returns distance of zero for same point

    reformat_indicies_x = np.repeat(np.arange(0, num_elements, 1), knn)
    reformat_indicies_y = np.reshape(indices, (-1))
    reformat_distances = np.reshape(distances, (-1)) / np.median(
        np.reshape(distances, (
            -1)))  # need to move this to the negative exponent of e talk about the bands
    std_indicies = np.std(distances, axis=1)
    std_2_per_distances = std_indicies[reformat_indicies_x] * std_indicies[
        reformat_indicies_y]
    reformat_distances_scaled = np.exp(
        -reformat_distances / std_2_per_distances)

    return sparse.csr_matrix(sparse.coo_matrix(
        (
            reformat_distances_scaled,
            (reformat_indicies_x, reformat_indicies_y)),
        shape=(num_elements, num_elements)))


def construct_D_inv(dim, K):
    """

    Parameters
    ----------
    dim dimensions for the graph
    K the sparse matrix K for the pairwise affinity

    Returns
    -------
    a sparse matrix with type csr, and D's diagnol values
    """
    # D_diag = np.nan_to_num(1/K.sum(axis=1), nan=0.0, posinf =0, neginf=0) #add small epsilon to each row in K.sum()
    D_diag = 1 / K.sum(axis=1)

    D_sparse = sparse.dia_matrix((np.reshape(D_diag, [1, -1]), [0]),
                                 (dim, dim))
    return sparse.csr_matrix(D_sparse), D_diag


def calc_laplacian(P_sparse, dim):
    I_sparse = sparse.identity(dim, format="csr")
    laplacian_sparse = I_sparse - P_sparse
    return laplacian_sparse


def calc_D_sqrt(D_diag, dim):
    D_sqrt = sparse.csr_matrix(
        sparse.dia_matrix((np.reshape(np.power(D_diag, .5), [1, -1]), [0]),
                          (dim, dim)))
    return D_sqrt


def calc_D_neg_sqrt(D_diag, dim):
    D_neg_sqrt = sparse.csr_matrix(
        sparse.dia_matrix(
            (np.reshape(np.power(D_diag, -.5), [1, -1]), [0]),
            (dim, dim)))
    return D_neg_sqrt


def embed_eigen(eigen_vectors):
    pixel_embedings = np.sum(np.power(eigen_vectors, 2),
                             axis=1)
    return pixel_embedings
