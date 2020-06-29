import os

from matplotlib import pyplot as plt

os.environ["OMP_NUM_THREADS"] = "10"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10"  # export NUMEXPR_NUM_THREADS=6
import scipy.sparse as sparse
import hnswlib as hnsw
import numpy as np
from typing import Tuple

from dask import delayed


@delayed
def calcAffinityMatrix(*, pixel_list: np.ndarray, metric: str, knn: int,
                       accuracy: int, connections: int, normalize_w_k: int,
                       num_threads:
                       int, spatial_box_num: int, temporal_box_num: int):
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
    normalize_w_k: kth clostest neighbor for autotune

    Returns
    -------
    affinity matrix
    """
    # TODO make connections value scale based on available memory

    dim = pixel_list.shape[1]
    num_elements = pixel_list.shape[0]
    print("Spatial Box {}, Time Step {}: Started Processing".format(spatial_box_num,
                                                                    temporal_box_num))
    knn_graph = hnsw.Index(space=metric, dim=dim)
    knn_graph.init_index(max_elements=num_elements, ef_construction=accuracy,
                         M=connections)
    knn_graph.add_items(pixel_list, num_threads=num_threads)
    indices, distances = knn_graph.knn_query(pixel_list, k=int(knn),
                                             num_threads=num_threads)
    # import faiss  # make faiss available
    # index = faiss.IndexFlatL2(dim)
    # pixel_list = np.ascontiguousarray(pixel_list.astype('float32'))
    # index.add(pixel_list)  # add vectors to the index
    # distances, indices = index.search(pixel_list, knn)
    # distances = np.power(distances,2)

    # lazy random walk means it returns distance of zero for same point
    # nbrs = LSHForest(n_estimators=20, n_candidates=200,
    #                  n_neighbors=10).fit(pixel_list)
    # distances, indices = nbrs.kneighbors(pixel_list)

    # from annoy import AnnoyIndex
    # t = AnnoyIndex(dim, 'dot')  # Length of item vector that will be indexed
    # for num,x in enumerate(pixel_list):
    #     t.add_item(num,x)
    # t.build(20)
    # indices = np.zeros((num_elements, knn), dtype=int)
    # distances = np.zeros((num_elements, knn))
    # for num in range(num_elements):
    #     indice, distance = t.get_nns_by_item(num, knn, search_k=-1, include_distances=True)
    #     indices[num] = indice
    #     distances[num] = distance
    # distances = np.power(distances,2)
    # indices = io.loadmat('/Users/sschickler/Code_Devel/LSSC-python/inputs/nf_0200_indices.mat')['Inds'].transpose((1, 0))-1
    # distances = io.loadmat('/Users/sschickler/Code_Devel/LSSC-python/inputs/nf_0200_distances.mat')['Dis'].transpose((1, 0))
    # print("Lodaed mat files")
    # print(indices.shape)
    # print(distances.shape)
    # print(num_elements)
    # TODO add comments here
    reformat_indicies_x = np.repeat(np.arange(0, num_elements, 1), knn)
    reformat_indicies_y = np.reshape(indices[:, 0:knn], (-1))
    reformat_distances = np.reshape(distances[:, 0:knn], (-1))

    # Self tuning adaptive bandwidth
    scale_factor_indices = np.repeat(distances[:, normalize_w_k], knn) + .000000001
    scale_factor_2_per_distances = np.power(scale_factor_indices[reformat_indicies_x],
                                            .5) * \
                                   np.power(scale_factor_indices[reformat_indicies_y],
                                            .5)
    reformat_distances_scaled = np.exp(
        -reformat_distances / scale_factor_2_per_distances)
    # TODO change to go direct to csr matrix
    reformat_distances_scaled[reformat_indicies_x == reformat_indicies_y] = 0
    K = sparse.csr_matrix(
        (
            reformat_distances_scaled,
            (reformat_indicies_x,
             reformat_indicies_y)),
        shape=(num_elements, num_elements))
    K_sym = (K + K.transpose()) / 2
    # K_sym[np.arange(0, num_elements, 1),np.arange(0, num_elements, 1)] = 0
    # save_image(K_sym.todense(),0)
    print(str(temporal_box_num)+" "+str(spatial_box_num) + "  " + str(
        np.count_nonzero(K_sym.diagonal()) / K_sym.diagonal().shape[0]))
    return K_sym


def save_image(image, num):
    print(image.diagonal())
    plt.imshow(
        image, aspect="auto")
    plt.colorbar()
    path = os.path.join("/Users/sschickler/Code_Devel/LSSC-python/input_images/test",
                        "eigen{}.png".format(
                            num))
    plt.savefig(path, dpi=300)
    plt.close()
    # image
    # img.save(image_path)
def calcDInv(K: sparse.csr_matrix):
    """Calculates a scaling diagonal matrix D to rescale eigen vectors

    Parameters
    ----------
    K the sparse matrix K for the pairwise affinity

    Returns
    -------
    a sparse matrix with type csr, and D's diagonal values
    """
    dim = K.shape[0]
    D_diag_inv = 1 / (
                K.sum(axis=1) + .000000001)  # add small epsilon to each row in K.sum()

    # D_diag = 1 / K.sum(axis=1)
    # print("D_diag",D_diag)
    D_sparse = sparse.dia_matrix((np.reshape(D_diag_inv, [1, -1]), [0]),
                                 (dim, dim))
    return sparse.csr_matrix(D_sparse), (K.sum(axis=1) + .000000001)


def calcLaplacian(P_sparse: sparse.csr_matrix, dim: Tuple[int, int]):
    """
    Calculates the Laplacian
    Parameters
    ----------
    P_sparse
        A sparse matrix of the normalized affinity matrix
    dim
        dimensions of P_sparse
    Returns
    -------
    Laplacian matrix
    # TODO I'm a little unsure what are the dimensions here
    """
    I_sparse = sparse.identity(dim, format="csr")
    laplacian_sparse = I_sparse - P_sparse
    return laplacian_sparse


def calcDSqrt(D_diag: np.ndarray):
    """
    Calculates sparse matrix of the sqrt of D
    Parameters
    ----------
    D_diag
        A list of the diagonal indices of D a the normalization matrix for P

    Returns
    -------
    A sparse CSR matrix of the sqrt of D
    """
    dim = D_diag.shape[0]
    D_sqrt = sparse.csr_matrix(
        sparse.dia_matrix((np.reshape(np.power(D_diag, .5), [1, -1]), [0]),
                          (dim, dim)))
    return D_sqrt


def calcDNegSqrt(D_diag):
    """
    Calculates sparse matrix of the neg sqrt of D
    Parameters
    ----------
    D_diag
        A list of the diagonal indices of D a the normalization matrix for P

    Returns
    -------
    A sparse CSR matrix of the neg sqrt of D
    """
    dim = D_diag.shape[0]
    D_neg_sqrt = sparse.csr_matrix(
        sparse.dia_matrix(
            (np.reshape(np.power(D_diag, -.5), [1, -1]), [0]),
            (dim, dim)))
    return D_neg_sqrt


def embedEigenSqrdNorm(eigen_vectors: np.ndarray) -> np.ndarray:
    """
    Embeds all pixels in image to a set of eigen vectors
    Parameters
    ----------
    eigen_vectors A set of eigen vectors

    Returns
    -------
    A list of pixel values if they where represented by those eigen vectors
    """
    pixel_embedings = np.sum(np.power(eigen_vectors, 2),
                             axis=1)
    return pixel_embedings
