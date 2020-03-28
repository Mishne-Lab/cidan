from LSSC.functions.embeddings import calc_D_sqrt, calc_D_neg_sqrt, calc_D_inv
import numpy as np
from dask import delayed
from scipy.sparse import linalg
import os
from PIL import Image
from LSSC.functions.pickle_funcs import pickle_save, pickle_exist, pickle_load
@delayed
def gen_eigen_vectors(*, K: np.ndarray, num_eig: int)->np.ndarray:
    """Calculate Eigen Vectors given parts of the affinity matrix

    Parameters
    ----------

    K
        An affinity matrix K for the image
    num_eig
        number of eigen values to generate

    Returns
    -------
    A matrix of eigen vectors
    """
    D_inv, D_diag = calc_D_inv(K=K)
    P = D_inv*K
    D_neg_sqrt = calc_D_neg_sqrt(D_diag)
    P_transformed = calc_D_sqrt(D_diag)*P*D_neg_sqrt
    eig_values, eig_vectors_scaled = linalg.eigsh(
        P_transformed, num_eig, which="LM",
        return_eigenvectors=True)
    eig_vectors = np.flip(
        D_neg_sqrt * eig_vectors_scaled, axis=1)
    return eig_vectors
@delayed
def save_eigen_vectors(*, e_vectors, spatial_box_num: int, time_box_num:int, save_dir:
str):
    eigen_dir = os.path.join(save_dir, "eigen_vectors")
    if not os.path.isdir(eigen_dir):
        os.mkdir(eigen_dir)
    pickle_save(e_vectors, name="eigen_vectors_box_{}_{"+
                                "}.pickle".format(spatial_box_num,time_box_num),
                output_directory=eigen_dir, )
    return e_vectors
@delayed
def load_eigen_vectors(*, spatial_box_num: int, time_box_num: int, save_dir: str):
    eigen_dir = os.path.join(save_dir, "eigen_vectors")
    return pickle_load(name="eigen_vectors_box_{}_{"
                                +"}.pickle".format(spatial_box_num,time_box_num),
                output_directory=eigen_dir)
@delayed
def save_embeding_norm_image(*, e_vectors, image_shape, save_dir, spatial_box_num):
    # print(save_dir)
    embed_dir = os.path.join(save_dir, "embedding_norm_images")
    e_vectors_squared = np.power(e_vectors, 2)
    e_vectors_sum = np.sum(e_vectors_squared, axis=1)

    e_vectors_sum_rescaled = e_vectors_sum * (
            15.0 / e_vectors_sum.max())  # add histogram equalization

    img = Image.fromarray(
        np.reshape(e_vectors_sum_rescaled,
                   image_shape[1:]) * 255).convert('L')
    image_path = os.path.join(embed_dir, "embedding_norm_image_box_{}.png".format(
        spatial_box_num))
    img.save(image_path)
    return e_vectors