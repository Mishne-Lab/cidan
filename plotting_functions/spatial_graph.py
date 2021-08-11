import json
import math
import os
import pickle
from os import listdir
from os.path import isfile, join

import fire
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


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
# data_list = [
# "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/kdan/KDA79_A_keep121.json",
# "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/kdan/roi_list.json"
# ]
# background_image_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/kdan/embedding_norm_image.png"
# eigen_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/neurofinder2.0"


def create_image_from_eigen_vectors(path, shape):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    vectors = []
    for x in onlyfiles:
        with open(os.path.join(path, x), "rb") as file:
            vectors.append(pickle.load(file)[:, 1:])
    all_vectors = np.hstack(vectors)
    all_vectors_sum = np.power(np.sum(np.power(all_vectors, 2), axis=1), .5)
    all_vectors_shaped = np.reshape(all_vectors_sum, shape)
    all_vectors_shaped[all_vectors_shaped < 0] = 0
    # if all_vectors_shaped.min()<0:
    #     all_vectors_shaped+=all_vectors_shaped.min()*-1
    return all_vectors_shaped * 255 / (all_vectors_shaped.max())


def extract_roi(image, roi_pixels, size, offset=0):
    image_new = np.zeros((image.shape[0] - offset, image.shape[1] - offset),
                         dtype="int")

    for pixel in roi_pixels:
        try:
            image_new[pixel[0], pixel[1]] = image[pixel[0] + offset, pixel[1] + offset]
        except IndexError:
            print("bad pixel location: " + str([x + offset for x in pixel]))
    roi = roi_pixels
    image_cropped = image_new[min([x[0] for x in roi]):max([x[0] for x in roi]) + 1,
                    min([x[1] for x in roi]):max([x[1] for x in roi]) + 1]
    cur_shape = image_cropped.shape
    diff = (size[0] - cur_shape[0], size[1] - cur_shape[1])
    image_padded = np.pad(image_cropped,
                          [[math.floor(diff[0] / 2), math.ceil(diff[0] / 2)],
                           [math.floor(diff[1] / 2), math.ceil(diff[1] / 2)]])
    return image_padded


def create_graph(bg_path="", shape=None, e_dir="", rois="", out_file="",
                 percent=99, pad=(0, 0),
                 offset=-10, roi_select=""):
    if bg_path != "":
        background_image = mpimg.imread(bg_path) * 200 / 255
        if pad[0] != 0:
            background_image = np.pad(background_image,
                                      [(pad[0], pad[0] + 1), (pad[1], pad[1] + 1)])
        # background_image = gaussian_filter(background_image, .02)
    if shape is None:
        shape = background_image.shape
    if e_dir != "":
        background_image = (255 / 255 * create_image_from_eigen_vectors(e_dir,
                                                                        shape))
    background_image = ((background_image / np.percentile(background_image, percent)))
    background_image[background_image > 1] = 1
    background_image = background_image * 200 + 55
    background_image[background_image < 0] = 0
    with open(rois, "r") as json_true:
        json_b_actual = json.load(json_true)
    roi_images = []
    roi_list = [x["coordinates"] for x in json_b_actual]
    if roi_select != "":
        roi_list = [roi_list[int(x)] for x in roi_select.split(" ")]
    size = (
    max([max([x[0] for x in roi]) - min([x[0] for x in roi]) for roi in roi_list]) + 5,
    max([max([x[1] for x in roi]) - min([x[1] for x in roi]) for roi in roi_list]) + 5)
    for num, x in enumerate(roi_list):
        cords = x
        roi_images.append(extract_roi(background_image, cords, size, offset))
    if len(roi_images) % 2 == 1:
        roi_images.pop()
    roi_display_img = np.vstack([np.hstack(roi_images[:len(roi_images) // 2]),
                                 np.hstack(roi_images[len(roi_images) // 2:])])
    roi_display_img = ((roi_display_img / np.percentile(roi_display_img, percent)))
    roi_display_img[roi_display_img > 1] = 1
    roi_display_img = roi_display_img * 220 + 35
    roi_display_img[roi_display_img < 0] = 0
    roi_display_img_combined = np.dstack(
        [np.zeros(roi_display_img.shape), roi_display_img,
         np.zeros(roi_display_img.shape)])

    plt.imsave(out_file, roi_display_img_combined.astype(np.uint8), vmin=0, vmax=255)


if __name__ == '__main__':
    fire.Fire(create_graph)
    """ --bg_path /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File1_CPn_l5_gcamp6s_lan.tif300/embedding_norm_images/embedding_norm_image.png 
    --rois /Users/sschickler/Code_Devel/LSSC-python/plotting_functions/data/cidan/File1_CPn_l5_gcamp6s_lan.tif300/roi_list.json
    --out_file plots/spatial_graph_file1.png
    """
    #  good rois in file 4 "0 2 7 15 16 22 23 24 25 26 28 32 33 35 37 42 43 47 48 49"
# 27 28 29 30 31 32 33 34 35 37 38 39 40 41"
