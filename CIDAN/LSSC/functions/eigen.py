from CIDAN.LSSC.functions.embeddings import calc_D_sqrt, calc_D_neg_sqrt, calc_D_inv
import numpy as np
from dask import delayed
from scipy.sparse import linalg
import os
from PIL import Image
from CIDAN.LSSC.SpatialBox import combine_images
from CIDAN.LSSC.functions.pickle_funcs import pickle_save, pickle_exist, pickle_load
import logging
logger1 = logging.getLogger("CIDAN.LSSC.eigen")
@delayed
def gen_eigen_vectors(*, K: np.ndarray, num_eig: int, spatial_box_num:int, temporal_box_num: int)->np.ndarray:
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
    # print("Spatial Box {}, Time Step {}: Finished ".format(spatial_box_num, temporal_box_num))

    return eig_vectors
@delayed
def save_eigen_vectors(*, e_vectors, spatial_box_num: int, time_box_num:int, save_dir:
str):
    eigen_dir = os.path.join(save_dir, "eigen_vectors")
    if not os.path.isdir(eigen_dir):
        os.mkdir(eigen_dir)
    pickle_save(e_vectors, name="eigen_vectors_box_{}_{}.pickle".format(spatial_box_num,time_box_num),
                output_directory=eigen_dir, )
    return e_vectors
@delayed
def load_eigen_vectors(*, spatial_box_num: int, time_box_num: int, save_dir: str):
    eigen_dir = os.path.join(save_dir, "eigen_vectors")
    name = "eigen_vectors_box_{}_{}.pickle".format(spatial_box_num,time_box_num)
    logger1.debug("Eigen vector load: file name: {0}, directory: {1}".format(name,eigen_dir))
    vector = pickle_load(name=name,
                output_directory=eigen_dir)
    logger1.debug("Eigen spatial box {0}, time box {1} shape: {2}".format(spatial_box_num, time_box_num, vector.shape))
    return vector
@delayed
def save_embeding_norm_image(*, e_vectors, image_shape, save_dir, spatial_box_num):
    # print(save_dir)
    # embed_dir = os.path.join(save_dir, "embedding_norm_images")
    # e_vectors_squared = np.power(e_vectors, 2)
    # e_vectors_sum = np.sum(e_vectors_squared, axis=1)
    #
    # e_vectors_sum_rescaled = e_vectors_sum * (
    #         10.0 / e_vectors_sum.max())  # add histogram equalization
    #
    # img = Image.fromarray(
    #     np.reshape(e_vectors_sum_rescaled,
    #                image_shape[1:]) * 255).convert('L')
    # image_path = os.path.join(embed_dir, "embedding_norm_image_box_{}.png".format(
    #     spatial_box_num))
    # img.save(image_path)
    return e_vectors
def create_embeding_norm_multiple(*,  spatial_box_list, save_dir, num_time_steps):
    # print(save_dir)
    eigen_dir = os.path.join(save_dir, "eigen_vectors")
    e_vectors_list = []
    for spatial_box in spatial_box_list:
        temp = []
        for time_box_num in range(num_time_steps):
            temp.append(pickle_load("eigen_vectors_box_{}_{}.pickle".format(spatial_box.box_num, time_box_num), output_directory=eigen_dir))
        e_vectors_list.append(np.hstack(temp))
    embed_dir = os.path.join(save_dir, "embedding_norm_images")
    eigen_images = []
    for e_vectors, spatial_box in zip(e_vectors_list, spatial_box_list):
        e_vectors_squared = np.power(e_vectors, 2)
        e_vectors_sum = np.sum(e_vectors_squared, axis=1)
        e_vectors_sum_rescaled = e_vectors_sum * (
                9.0 / e_vectors_sum.max())  # add histogram equalization
        e_vectors_shaped = np.reshape(e_vectors_sum,
                   spatial_box.shape[1:])
        eigen_images.append(e_vectors_shaped)
    image = combine_images(spatial_box_list,eigen_images)



    img = Image.fromarray(image * (
                3.0 / image.max())*255).convert('L')
    image_path = os.path.join(embed_dir, "embedding_norm_image.png")
    img.save(image_path)
    return e_vectors_list