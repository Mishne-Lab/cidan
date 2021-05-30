import logging
import os

import numpy as np
from PIL import Image
from dask import delayed
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg

from cidan.LSSC.SpatialBox import combine_images
from cidan.LSSC.functions.embeddings import calcDInv, calcDSqrt, calcDNegSqrt
from cidan.LSSC.functions.pickle_funcs import pickle_save, pickle_load

logger1 = logging.getLogger("cidan.LSSC.eigen")


@delayed
def generateEigenVectors(*, K: sparse.csr_matrix, num_eig: int, maxiter=7,
                         accuracy=1E-4) -> np.ndarray:
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
    D_inv, D_diag = calcDInv(K=K)
    P = D_inv.dot(K)
    D_neg_sqrt = calcDNegSqrt(D_diag)
    P_transformed = calcDSqrt(D_diag).dot(P).dot(D_neg_sqrt)
    # print("Start eigen", num_eig)
    # print(accuracy)
    # eig_values,eig_vectors = eig(P.todense())
    # io.savemat("C:\\Users\\gadge\\Documents\\cidan\\inputs\\P.mat", {"array": P)}

    eig_values, eig_vectors_scaled = linalg.eigsh(
        P_transformed, num_eig, which="LM",
        return_eigenvectors=True, maxiter=maxiter * num_eig,
        tol=accuracy)  # this returns normalized eigen vectors
    # print("finished eigen")
    # # TODO make first eigen vector be sanity check since all elements are the same
    # #  this isn't the case
    # # print("Eigvalues",eig_values[0], eig_vectors_scaled,np.max(eig_vectors_scaled),eig_vectors_scaled.shape, num_eig)
    eig_vectors = np.flip(
        D_neg_sqrt.dot(eig_vectors_scaled),
        axis=1)  # this preforms matrix multiplication

    return np.real(eig_vectors)[:, 1:]


@delayed
def saveEigenVectors(*, e_vectors, spatial_box_num: int, time_box_num: int, save_dir:
str, total: int):

    eigen_dir = os.path.join(save_dir, "eigen_vectors")
    if not os.path.isdir(eigen_dir):
        os.mkdir(eigen_dir)
    pickle_save(e_vectors, name="eigen_vectors_box_{}_{}.pickle".format(spatial_box_num,
                                                                        time_box_num),
                output_directory=eigen_dir, )
    # print("save Eigen done", time_box_num, " ", spatial_box_num)

    return e_vectors


@delayed
def loadEigenVectors(*, spatial_box_num: int, time_box_num: int, save_dir: str):
    eigen_dir = os.path.join(save_dir, "eigen_vectors")
    name = "eigen_vectors_box_{}_{}.pickle".format(spatial_box_num, time_box_num)
    logger1.debug(
        "Eigen vector load: file name: {0}, directory: {1}".format(name, eigen_dir))
    vector = pickle_load(name=name,
                         output_directory=eigen_dir)
    logger1.debug(
        "Eigen spatial box {0}, time box {1} shape: {2}".format(spatial_box_num,
                                                                time_box_num,
                                                                vector.shape))  # noqa
    return vector


@delayed
def saveEmbedingNormImage(*, e_vectors, image_shape, save_dir, spatial_box_num):
    # print("Eigen norm step")
    embed_dir = os.path.join(save_dir, "embedding_norm_images")
    e_vectors_squared = np.power(e_vectors, 2)
    e_vectors_sum = np.sum(e_vectors_squared, axis=1)
    e_vectors_sum = np.power(e_vectors_sum, .5)

    e_vectors_sum_rescaled = e_vectors_sum * (
            10.0 / e_vectors_sum.mean())  # add histogram equalization

    img = Image.fromarray(
        np.reshape(e_vectors_sum_rescaled,
                   image_shape) * 255).convert('L')
    image_path = os.path.join(embed_dir, "embedding_norm_image_box_{}.png".format(
        spatial_box_num))
    img.save(image_path)
    return e_vectors


def createEmbedingNormImageFromMultiple(*, spatial_box_list, save_dir, num_time_steps):
    """
    This function takes in a place where eigen vectors are stored and creates a full
    image for the entire image instead of just each spatial box
    Parameters
    ----------
    spatial_box_list list of spatial boxes
    save_dir where eigen vectors are stored
    num_time_steps number of timesteps used

    Returns
    -------

    """
    eigen_dir = os.path.join(save_dir, "eigen_vectors")
    e_vectors_list = []
    for spatial_box in spatial_box_list:
        temp = []
        for time_box_num in range(num_time_steps):
            e_vectors = pickle_load(
                "eigen_vectors_box_{}_{}.pickle".format(spatial_box.box_num,
                                                        time_box_num),
                output_directory=eigen_dir)

            fig = plt.figure(frameon=False)
            for x in range(e_vectors.shape[1]):
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(np.abs(e_vectors[:, x].reshape(spatial_box.shape)))
                fig.savefig(os.path.join(save_dir, "embedding_norm_images",
                                         "e_vector2_%s.png" % (str(x))))
            fig.clf()
            e_vectors_squared = np.power(e_vectors, 2)
            e_vectors_sum = np.sum(e_vectors_squared, axis=1)
            e_vectors_sum = np.power(e_vectors_sum, .5)

            temp.append(e_vectors_sum)

        e_vectors_list.append(np.max(temp, axis=0))
    embed_dir = os.path.join(save_dir, "embedding_norm_images")
    eigen_images = []
    for e_vectors, spatial_box in zip(e_vectors_list, spatial_box_list):
        # e_vectors_sum_rescaled = e_vectors * (
        #         9.0 / e_vectors.mean())  # add histogram equalization
        # noqa
        e_vectors_shaped = np.reshape(e_vectors,
                                      spatial_box.shape)
        eigen_images.append(e_vectors_shaped)
    image = combine_images(spatial_box_list, eigen_images)
    percent_95 = np.percentile(image, 95.0)
    percent_05 = np.percentile(image, 5.0)

    img = Image.fromarray(
        ((image - percent_05) / (percent_95 - percent_05)) * 255).convert('L')
    image_path = os.path.join(embed_dir, "embedding_norm_image.png")
    img.save(image_path)

    # plt.imshow(img)
    # plt.savefig(os.path.join(embed_dir, "embedding_norm_matlab.png"))
    # plt.close()

    return e_vectors_list


def scale_background(background_image):
    background_image[background_image < 0] = 0

    background_image = (((background_image - np.percentile(background_image, 1)) / (
            np.percentile(background_image, 99) - np.percentile(
        background_image, 1))))

    background_image[background_image > 1] = 1
    background_image = background_image * 255
    background_image[background_image < 0] = 0
    return background_image
