import os
import pickle
from os import listdir
from os.path import isfile, join

import fire
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

colors = [(255, 10, 10), (255, 200, 15)]
overlap = (230, 66, 24)
# data_list = [
#     "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/demo_files/roi_list1.json",
#     "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/demo_files/roi_list2.json"]
# background_image_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/demo_files/embedding_norm_image.png"
# eigen_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/demo_files/eigen_vectors"
# data_list = [
#     "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/kdan/KDA79_A_keep121.json",
#     "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/kdan/roi_list.json"
# ]
# background_image_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/kdan/embedding_norm_image.png"
# eigen_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/kdan/eigen_vectors"
data_list = [
    # "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/kdan/KDA79_A_keep121.json",
    # "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/kdan/roi_list.json"
]
# background_image_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/kdan/embedding_norm_image.png"
eigen_path = "/home/sschickl/Desktop/File5_l23_gcamp6s_lan.tif330/eigen_vectors/"
use_eigen_background = True


def create_image_from_eigen_vectors(path, shape):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    vectors = []
    for x in onlyfiles:
        with open(os.path.join(path, x), "rb") as file:
            vectors.append(pickle.load(file)[:, 1:])
    all_vectors = np.hstack(vectors)
    # savemat("test" + ".mat", {"data": all_vectors},
    #         appendmat=True)
    all_vectors_sum = np.power(np.sum(np.power(all_vectors, 2), axis=1), .5)
    all_vectors_shaped = np.reshape(all_vectors_sum, shape)
    all_vectors_shaped[all_vectors_shaped < 0] = 0
    # if all_vectors_shaped.min()<0:
    #     all_vectors_shaped+=all_vectors_shaped.min()*-1
    return all_vectors_shaped * 255 / (all_vectors_shaped.max())

def create_images_from_eigen_vectors(path, shape):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    vectors = []
    for x in onlyfiles:
        with open(os.path.join(path, x), "rb") as file:
            vector = pickle.load(file)[:, 1:]
        all_vectors_sum = np.power(np.sum(np.power(vector, 2), axis=1), .5)
        all_vectors_shaped = np.reshape(all_vectors_sum, shape)
        all_vectors_shaped[all_vectors_shaped < 0] = 0
        vectors.append(all_vectors_shaped)
    # if all_vectors_shaped.min()<0:
    #     all_vectors_shaped+=all_vectors_shaped.min()*-1
    return vectors
def display_eigen(e_dir=eigen_path, out_file="test", shape=[235,235], percent=99, many=False):
    """

    Parameters
    ----------
    e_dir : str
        test
    out_file
    shape
    percent

    Returns
    -------

    """
    if many:
        percent = int(percent)
        vectors = create_images_from_eigen_vectors(e_dir,shape)
        for num,x in enumerate(vectors):
            background_image = (255 / 255 * x)
            # background_image = gaussian_filter(background_image,1)
            background_image = ((background_image / np.percentile(background_image, percent)))
            background_image[background_image > 1] = 1
            background_image = background_image * 200 + 55
            background_image[background_image < 0] = 0

            combined_image = np.dstack([np.zeros(shape), background_image,
                                        np.zeros(shape)])
            print(out_file[-4:]+"_%s.png"%str(num))
            plt.imsave(out_file[:-4]+"_%s.png"%str(num), combined_image.astype(np.uint8))

    percent = int(percent)
    background_image = (255 / 255 * create_image_from_eigen_vectors(e_dir,
                                                                    shape))
    # background_image = gaussian_filter(background_image,1)
    background_image = ((background_image / np.percentile(background_image, percent)))
    background_image[background_image > 1] = 1
    background_image = background_image * 200 + 55
    background_image[background_image < 0] = 0

    combined_image = np.dstack([np.zeros(shape), background_image,
                                np.zeros(shape)])

    plt.imsave(out_file, combined_image.astype(np.uint8))


if __name__ == '__main__':
    fire.Fire(display_eigen)
