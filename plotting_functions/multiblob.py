import json
import os
import pickle
from os import listdir
from os.path import isfile, join

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

colors = [(255, 0, 0), (255, 200, 15)]
overlap = (230, 66, 24)
data_list = [
    "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/demo_files/roi_list1.json",
    "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/demo_files/roi_list2.json"]
background_image_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/demo_files/embedding_norm_image.png"
eigen_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/demo_files/eigen_vectors"
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


def create_roi_image(size, color, path):
    image = np.zeros((size[0], size[1], 3), dtype="int")

    with open(path, "r") as json_true:
        json_b_actual = json.load(json_true)
    for num, x in enumerate(json_b_actual):
        cords = x["coordinates"]
        for pixel in cords:
            image[pixel[0] - 1, pixel[1] - 1] = color
    return image


background_image = mpimg.imread(background_image_path) * 200 / 255
background_image = gaussian_filter(background_image, .02)
background_image = (255 / 255 * create_image_from_eigen_vectors(eigen_path,
                                                                background_image.shape))
# background_image = gaussian_filter(background_image,1)
background_image = ((background_image / np.percentile(background_image, 99)))
background_image[background_image > 1] = 1
background_image = background_image * 200 + 55
background_image[background_image < 0] = 0

combined_image = np.dstack([np.zeros(background_image.shape), background_image,
                            np.zeros(background_image.shape)])
roi_image_combined = np.dstack(
    [np.zeros(background_image.shape), np.zeros(background_image.shape),
     np.zeros(background_image.shape)])
roi_image_combined_single = np.dstack(
    [np.zeros(background_image.shape), np.zeros(background_image.shape),
     np.zeros(background_image.shape)])
for x, color in zip(data_list, colors):
    roi_image = create_roi_image(background_image.shape, color, x)
    roi_image_single = create_roi_image(background_image.shape, (1, 0, 0), x)
    roi_image_combined += roi_image
    roi_image_combined_single += roi_image_single
roi_image_sum = np.sum(roi_image_combined, axis=2)
roi_image_combined_single_sum = np.sum(roi_image_combined_single, axis=2)

# combined_image[roi_image_sum!=0]=[0,0,0]
# combined_image+=roi_image_combined
# combined_image[roi_image_combined_single_sum>1]=overlap
combined_image = combined_image.astype(int)
imgplot = plt.imshow(combined_image)
plt.show()
