from LSSC.functions import data_manipulation, embeddings
from LSSC.functions.pickle_funcs import *
from LSSC.functions.clustering import *
from scipy.sparse import linalg
import numpy as np
from PIL import Image
from skimage import measure

from LSSC.Parameters import Parameters
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# from IPython.display import display, Image


# TODO  DONE add slice function

# TODO DONE add infill path
# TODO add motion registration causes artifacts at edges
# TODO add can sometime see window because window comes out as 1 giant connected component
# TODO add something that throws out lines using Eccintricity heuristics
# row/column sum on max projection image


class Stack:
    """
    This is a class to interact with different operations on a stack of images.
    It is lazy, so it won't load any data or do anything else until necessary
    Each function is designed to produce the necessary values with the least
    computation possible. It will first try to load it from memory, then from
    the disk, then only if both of those fail does it generate the information
    """

    def __init__(self, file_path, trial_index, output_directory, parameters,
                 save_images=False, gen_new=False):
        """
        Initializes a stack object
        Parameters
        ----------
        file_path path to the tif stack
        trial_index index of trial number
        parameters parameter object that contains info for clustering/matrix ops
        gen_new whether to load already generated objects from disk
        """
        self.file_path = file_path
        self.trial_index = trial_index
        self.gen_new = gen_new
        self.parameters = parameters
        self.save_images = save_images
        self.priv_volume = False
        self.priv_volume_2d = False
        self.priv_K = False
        self.priv_D_inv = False
        self.priv_D_diag = False
        self.priv_D_sqrt = False
        self.priv_D_neg_sqrt = False
        self.priv_P = False
        self.priv_laplacian = False
        self.priv_eig_values = False
        self.priv_eig_vectors = False

        self.priv_embeding = False  # 2d embeding of pixels in embeding space
        self.output_dir = os.path.join(output_directory, str(trial_index))
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        pickle_save(Parameters, "parameters", output_directory=self.output_dir)
        if self.gen_new:
            pickle_clear(trial_num=self.trial_index)

    def volume(self, vol=False):
        """
        Loads the volume and does the original filters, slicing and bounding box
        Parameters
        ----------
        vol: used if vol is already loaded, mainly by Stack_Wrapper

        Returns
        -------
        3D volume but also sets self.priv_volume to correct volume
        """
        if type(self.priv_volume) != bool:
            return self.priv_volume
        if not vol:
            self.priv_volume = data_manipulation.load_tif_stack(self.file_path)
        else:
            self.priv_volume = vol

        if self.parameters.slice:
            self.priv_volume = self.priv_volume[
                               self.parameters.slice_start::self.parameters.slice_every,
                               :, :]
        if self.parameters.bounding_box:
            box = self.parameters.bounding_box_val
            self.priv_volume=self.priv_volume[box[0][2]:box[1][2], box[0][0]:box[1][0], box[0][1]:box[1][1]]
        if self.save_images:
            data_manipulation.save_image(self.priv_volume, "original_image",
                                         self.output_dir,
                                         self.priv_volume.shape,
                                         number_save=4)
            data_manipulation.save_image(np.max(self.priv_volume, axis=0),
                                         "max_intensity",
                                         self.output_dir, (
                                             1, self.priv_volume.shape[1],
                                             self.priv_volume.shape[2]),
                                         number_save=1)
            data_manipulation.save_image(np.mean(self.priv_volume, axis=0),
                                         "mean_intensity",
                                         self.output_dir,
                                         (1, self.priv_volume.shape[1],
                                          self.priv_volume.shape[2]),
                                         number_save=1)
        if self.parameters.filter:
            self.priv_volume = data_manipulation.filter_stack(self.priv_volume,
                                                              self.parameters)
            if self.save_images:
                data_manipulation.save_image(self.priv_volume,
                                             "original_image_filtered",
                                             self.output_dir,
                                             self.priv_volume.shape,
                                             number_save=4)
                data_manipulation.save_image(np.max(self.priv_volume, axis=0),
                                             "max_intensity_filtered",
                                             self.output_dir, (
                                                 1, self.priv_volume.shape[1],
                                                 self.priv_volume.shape[2]),
                                             number_save=1)
                data_manipulation.save_image(np.mean(self.priv_volume, axis=0),
                                             "mean_intensity_filtered",
                                             self.output_dir,
                                             (1, self.priv_volume.shape[1],
                                              self.priv_volume.shape[2]),
                                             number_save=1)
        return self.priv_volume

    def volume_2d(self):
        if type(self.priv_volume_2d) != bool:
            return self.priv_volume_2d
        self.priv_volume_2d = data_manipulation.reshape_to_2d_over_time(
            self.volume())
        return self.priv_volume_2d

    def pixel_length(self):
        return self.volume().shape[1] * self.volume().shape[2]

    def original_shape(self):
        return self.volume().shape

    def K(self):
        """
        Calculates affinity matrix K
        Returns
        -------

        """
        if type(self.priv_K) != bool:
            return self.priv_K
        elif pickle_exist("K", trial_num=self.trial_index):
            self.priv_K = pickle_load("K", trial_num=self.trial_index)
        else:
            self.priv_K = embeddings.calc_affinity_matrix(self.volume_2d(),
                                                          parameters=self.parameters)
        return self.priv_K

    def D_inv(self):
        """
        Calculates the inverse of diagonal matrix D, D is just the diagonal of
        all the eigen vectors with everything else set to 0
        Returns
        -------
        D_inv (2d np.array)
        """
        if type(self.priv_D_inv) != bool:
            return self.priv_D_inv
        elif pickle_exist("D_inv", trial_num=self.trial_index):
            self.priv_D_inv = pickle_load("D_inv", trial_num=self.trial_index)
        else:
            self.priv_D_inv, self.priv_D_diag = embeddings.construct_D_inv(
                self.pixel_length(),
                self.K())
            pickle_save(self.priv_D_inv, "D_inv", trial_num=self.trial_index)
            pickle_save(self.priv_D_diag, "D_diag", trial_num=self.trial_index)
        return self.priv_D_inv

    def D_diag(self):
        """
        Calculates the diagonal of matrix D, this is just a diagonal of all
        the eigen vectors with everything else set to 0
        Returns
        -------
        D_diag(1d np array)
        """
        if type(self.priv_D_diag) != bool:
            return self.priv_D_diag
        elif pickle_exist("D_diag", trial_num=self.trial_index):
            self.priv_D_diag = pickle_load("D_diag",
                                           trial_num=self.trial_index)
        else:
            self.priv_D_inv, self.priv_D_diag = embeddings.construct_D_inv(
                self.pixel_length(),
                self.K())
            pickle_save(self.priv_D_inv, "D_inv", trial_num=self.trial_index)
            pickle_save(self.priv_D_diag, "D_diag", trial_num=self.trial_index)
        return self.priv_D_diag

    def P(self):
        """
        Calculates probability matrix P
        Returns
        -------
        P 2d np array
        """
        if type(self.priv_P) != bool:
            return self.priv_P
        else:
            self.priv_P = self.D_inv() * self.K()
        return self.priv_P

    def laplacian(self):
        """
        Calculates laplacian-not currently used in algorithm
        Returns
        -------
        laplacian as 2d array
        """
        if type(self.priv_laplacian) != bool:
            return self.priv_laplacian
        else:
            self.priv_laplacian = embeddings.calc_laplacian(self.P(),
                                                            self.pixel_length())
        return self.priv_laplacian

    def D_sqrt(self):
        """
        Calculate d_diag with a sqrt
        Returns
        -------
        2D array of d_diag sqrt
        """
        if type(self.priv_D_sqrt) != bool:
            return self.priv_D_sqrt
        else:
            self.priv_D_sqrt = embeddings.calc_D_sqrt(self.D_diag(),
                                                      self.pixel_length())
        return self.priv_D_sqrt

    def D_neg_sqrt(self):
        """
        Calculate d_diag with a negative sqrt
        Returns
        -------
        2D array of d_diag neg sqrt
        """
        if type(self.priv_D_neg_sqrt) != bool:
            return self.priv_D_neg_sqrt
        else:
            self.priv_D_neg_sqrt = embeddings.calc_D_neg_sqrt(self.D_diag(),
                                                              self.pixel_length())
        return self.priv_D_neg_sqrt

    def eigs(self):
        """
        Calculates Eigen vectors
        Returns
        -------
        Eigen_values(1d np array), eigen_vectors (2d np array)
        """
        if type(self.priv_eig_values) != bool and type(
                self.priv_eig_vectors) != bool:
            return self.priv_eig_values, self.priv_eig_vectors
        elif pickle_exist("eig_values",
                          trial_num=self.trial_index) and pickle_exist(
            "eig_vectors", trial_num=self.trial_index):
            self.priv_eig_values = pickle_load("eig_values",
                                               trial_num=self.trial_index)
            self.priv_eig_vectors = pickle_load("eig_vectors",
                                                trial_num=self.trial_index)

        else:
            P_transformed = self.D_sqrt() * self.P() * self.D_neg_sqrt()
            self.priv_eig_values, eig_vectors_scaled = linalg.eigsh(
                P_transformed, self.parameters.num_eig, which="LM",
                return_eigenvectors=True)
            self.priv_eig_vectors = np.flip(
                self.D_neg_sqrt() * eig_vectors_scaled, axis=1)

            pickle_save(self.priv_eig_values, "eig_values",
                        trial_num=self.trial_index)
            pickle_save(self.priv_eig_vectors, "eig_vectors",
                        trial_num=self.trial_index)
            if True:
                e_vectors_squared = np.power(self.priv_eig_vectors, 2)

                e_vectors_reshape = np.transpose(
                    np.reshape(e_vectors_squared, (
                        self.original_shape()[1],
                        self.original_shape()[2], self.parameters.num_eig,),
                               order="C"), (2, 0, 1))
                data_manipulation.save_image(e_vectors_reshape,
                                             "eigen_vectors", self.output_dir,
                                             e_vectors_reshape.shape,
                                             number_save=self.parameters.num_eig)
        return self.priv_eig_values, self.priv_eig_vectors

    def embeding_image(self, image_path):
        e_vectors = self.eigs()[1]
        e_vectors_squared = np.power(e_vectors, 2)
        e_vectors_sum = np.sum(e_vectors_squared, axis=1)

        e_vectors_sum_rescaled = e_vectors_sum * (
                15.0 / e_vectors_sum.max())  # add histogram equalization

        img = Image.fromarray(
            np.reshape(e_vectors_sum_rescaled,
                       self.original_shape()[1:]) * 255).convert('L')
        img.save(image_path)

    def clusters(self):
        # TODO Done take all eigen vectors that have embedings for pixel that are at least t percent of the biggest like 10%
        e_vectors = self.eigs()[1][:, 1:]
        cluster_list = cluster_image(e_vectors,
                                     original_2d_vol=self.volume_2d(),
                                     original_shape=self.original_shape(),
                                     parameters=self.parameters)
        if self.save_images:
            original_zeros_all = np.zeros((self.pixel_length()))
            for num, x in enumerate(cluster_list):
                original_zeros = np.zeros((self.pixel_length()))
                original_zeros_all[x] = 255
                original_zeros[x] = 255
                data_manipulation.save_image(original_zeros,
                                             "cluster_{}".format(num),
                                             self.output_dir, (
                                                 1, self.original_shape()[1],
                                                 self.original_shape()[2]), 1)

            data_manipulation.save_image(original_zeros_all, "clusterall",
                                         self.output_dir, (
                                             1, self.original_shape()[1],
                                             self.original_shape()[2]), 1)
        return cluster_list


if __name__ == '__main__':
    for x in range(71,80):
        data_stack = Stack(
            file_path="/data2/Sam/pythonTestEnviroment/input_images/test_stack.tif",
            trial_index=x,
            output_directory="/data2/Sam/pythonTestEnviroment/output_images/",
            save_images=True,
            parameters=Parameters(num_threads=10, knn=50, num_eig=50,
                                  median_filter=True, median_filter_size=(1,x-70,x-70),z_score=True), gen_new=True,
        )
        data_stack.clusters()

    # TODO DONE add in slicing of dataset
    # data_stack = Stack(
    #     file_path="/data2/Sam/pythonTestEnviroment/input_images/8_6_14_d10_001.tif",
    #     trial_index=30,
    #     output_directory="/data2/Sam/pythonTestEnviroment/output_images/",
    #     save_images=True,
    #     parameters=Parameters(num_threads=10, knn=300, num_eig=300,
    #                           accuracy=400, connections=400, slice_stack=True,
    #                           slice_every=3, slice_start=0, median_filter=True,
    #                           z_score=True), gen_new=False)
    # data_stack = Stack(
    #     file_path="/Users/sschickler/Documents/LSSC-python/input_images/" +
    #               "small_dataset.tif",
    #     trial_index=3,
    #     output_directory="/data2/Sam/pythonTestEnviroment/output_images/",
    #     parameters=Parameters(num_threads=70, knn=300, num_eig=300,
    #                           accuracy=400,
    #                           connections=400), gen_new=False)

