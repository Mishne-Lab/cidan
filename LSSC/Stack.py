from LSSC.functions import data_manipulation, embeddings
from LSSC.functions.pickle_funcs import *
from LSSC.functions.clustering import *
from scipy.sparse import linalg
import numpy as np
from PIL import Image
from skimage import measure
from LSSC.Parameters import Parameters


# TODO  add slice function
# TODO add cluster refinement
# TODO add infill path
# TODO add motion registration causes artifacts at edges
# TODO add can sometime see window because window comes out as 1 giant connected component
# TODO add something that throws out lines using Eccintricity heuristics


class Stack:
    def __init__(self, file_path, trial_index, parameters, gen_new=False):
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
        self.gen_new = False
        self.parameters = parameters
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

    def volume(self):
        if type(self.priv_volume) != bool:
            return self.priv_volume
        self.priv_volume = data_manipulation.load_tif_stack(self.file_path)
        return self.priv_volume

    def volume_2d(self):
        if type(self.priv_volume_2d) != bool:
            return self.priv_volume_2d
        self.priv_volume_2d = data_manipulation.reshape_to_2d_over_time(
            self.priv_volume)
        return self.priv_volume_2d

    def pixel_length(self):
        return self.volume().shape[1] * self.volume().shape[2]

    def original_shape(self):
        return self.volume().shape

    def K(self):
        if type(self.priv_K) != bool:
            return self.priv_K
        elif pickle_exist("K", trial_num=self.trial_index):
            self.priv_K = pickle_load("K", trial_num=self.trial_index)
        else:
            self.priv_K = embeddings.calc_affinity_matrix(self.volume_2d(),
                                                          metric=self.parameters.metric,
                                                          knn=self.parameters.knn,
                                                          accuracy=self.parameters.accuracy,
                                                          connections=self.parameters.connections,
                                                          num_threads=self.parameters.num_threads)
        return self.priv_K

    def D_inv(self):
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
        if type(self.priv_P) != bool:
            return self.priv_P
        else:
            self.priv_P = self.D_inv() * self.K()
        return self.priv_P

    def laplacian(self):
        if type(self.priv_laplacian) != bool:
            return self.priv_laplacian
        else:
            self.priv_laplacian = embeddings.calc_laplacian(self.P(),
                                                            self.pixel_length())
        return self.priv_laplacian

    def D_sqrt(self):
        if type(self.priv_D_sqrt) != bool:
            return self.priv_D_sqrt
        else:
            self.priv_D_sqrt = embeddings.calc_D_sqrt(self.D_diag(),
                                                      self.pixel_length())
        return self.priv_D_sqrt

    def D_neg_sqrt(self):
        if type(self.priv_D_neg_sqrt) != bool:
            return self.priv_D_neg_sqrt
        else:
            self.priv_D_neg_sqrt = embeddings.calc_D_neg_sqrt(self.D_diag(),
                                                              self.pixel_length())
        return self.priv_D_neg_sqrt

    def eigs(self):
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
            # TODO are eigen vectors also scaled if take decomp from D.5*P
            P_transformed = self.D_sqrt() * self.P() * self.D_neg_sqrt()
            self.priv_eig_values, eig_vectors_scaled = linalg.eigsh(
                P_transformed, self.parameters.num_eig, which="LM",
                return_eigenvectors=True)
            self.priv_eig_vectors = self.D_neg_sqrt() * eig_vectors_scaled
            pickle_save(self.priv_eig_values, "eig_values",
                        trial_num=self.trial_index)
            pickle_save(self.priv_eig_vectors, "eig_vectors",
                        trial_num=self.trial_index)
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

    def clusters(self, num_clusters, refinement=False,
                 num_eigen_vector_select=5):
        # TODO take all eigen vectors that have embedings for pixel that are at least t percent of the biggest like 10%
        e_vectors = self.eigs()[1]


        pixels_in_cluster = np.array([], dtype=np.int32)

        for num, x in enumerate(cluster_list):
            original_zeros = np.zeros((self.pixel_length()))

            original_zeros[x] = 255
            img = Image.fromarray(
                np.reshape(original_zeros,
                           self.original_shape()[1:])).convert('L')
            img.save("/data2/Sam/pythonTestEnviroment/output_images/cluster_tests_with_component/my%i.png" % num)


data_stack = Stack(
    "/data2/Sam/pythonTestEnviroment/input_images/8_6_14_d10_001.tif", 1,
    Parameters(num_threads=10))
data_stack.clusters(80)
