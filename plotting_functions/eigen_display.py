import os
import pickle
from os import listdir
from os.path import isfile, join

import fire
import matplotlib.pyplot as plt
import numpy as np

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
eigen_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/neurofinder2.0"
use_eigen_background = True


def create_image_from_eigen_vectors(path, shape):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    vectors = []
    for x in onlyfiles:
        with open(os.path.join(path, x), "rb") as file:
            vectors.append(pickle.load(file)[:, 1:])
    all_vectors = np.hstack(vectors)
    all_vectors_sum = np.sum(np.abs(all_vectors), axis=1)
    all_vectors_shaped = np.reshape(all_vectors_sum, shape)
    all_vectors_shaped[all_vectors_shaped < 0] = 0
    # if all_vectors_shaped.min()<0:
    #     all_vectors_shaped+=all_vectors_shaped.min()*-1
    return all_vectors_shaped * 255 / (all_vectors_shaped.max())


def display_eigen(e_dir, out_file, shape, percent=99):
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
