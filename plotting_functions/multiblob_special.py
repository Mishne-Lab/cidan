import json
import os
import pickle
from os import listdir
from os.path import isfile, join

import fire
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from skimage import feature
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
from skimage.measure import find_contours


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


def create_roi_image(size, color, path, blobs=True, offset=0):
    image = np.zeros((size[0], size[1], 3), dtype="int")

    with open(path, "r") as json_true:
        json_b_actual = json.load(json_true)
    for num, x in enumerate(json_b_actual):

        cords = x["coordinates"]
        if len(cords) < 600:
            image_temp = np.zeros((size[0], size[1], 3), dtype="int")

            for pixel in cords:
                try:
                    image_temp[pixel[0] + offset, pixel[1] + offset] = color
                except IndexError:
                    print("bad pixel location: " + str([x + offset for x in pixel]))
            if not blobs:
                edge = feature.canny(
                    np.sum(image_temp, axis=2) / np.max(np.sum(image_temp, axis=2)))
                image[edge] = color
            else:
                image[image_temp != 0] = image_temp[image_temp != 0]
    return image


def create_roi_image_cont(size, color, path, fig, offset=0, ):
    image = np.zeros((size[0], size[1], 3), dtype="int")

    with open(path, "r") as json_true:
        json_b_actual = json.load(json_true)
    for num, x in enumerate(json_b_actual):
        image = np.zeros((size[0], size[1]), dtype=float)

        cords = x["coordinates"]
        if len(cords) < 600:
            image_temp = np.zeros((size[0], size[1]), dtype=float)

            for pixel in cords:
                try:
                    image_temp[pixel[0] + offset, pixel[1] + offset] = 1
                except IndexError:
                    print("bad pixel location: " + str([x + offset for x in pixel]))
            image = np.zeros((size[0], size[1]), dtype=float)
            # edge = feature.canny(
            #     np.sum(image_temp, axis=2) / np.max(np.sum(image_temp, axis=2)))
            # image[edge] = 1
            image_temp = ndimage.morphology.binary_dilation(image_temp)
            # image_temp = ndimage.morphology.binary_dilation(image_temp)
            # image_temp = ndimage.morphology.binary_erosion(image_temp)
            #
            # image_temp = ndimage.morphology.binary_erosion(image_temp)
            # image_temp = ndimage.binary_closing(image_temp)
            contour = find_contours(image_temp, .3)
            print(contour[0][:, 0])
            # plt.imshow(image)
            fig.plot(contour[0][:, 1], contour[0][:, 0], color=color, linewidth=2)
    return image


def create_graph(bg_path="", shape=None, e_dir="", data_1="", data_2="", data_3="",
                 data_4="", out_file="",
                 percent=97, blobs=True, pad=(0, 0),
                 color_1=(234, 32, 39), color_2=(247, 159, 31), color_3=(6, 82, 221),
                 color_4=(217, 128, 250), overlap_c=(211, 84, 0),
                 offset=0):
    if bg_path != "":
        if ".npy" in bg_path:
            background_image = np.load(bg_path)
            print(0, background_image.shape)
        else:
            background_image = mpimg.imread(bg_path) * 200 / 255
            if pad[0] != 0:
                background_image = np.pad(background_image,
                                          [(pad[0], pad[0] + 1), (pad[1], pad[1] + 1)])
        # background_image = gaussian_filter(background_image, .02)
    print(shape)
    if shape is None:
        shape = background_image.shape
        print(1, background_image.shape)

    if e_dir != "":
        background_image = (255 / 255 * create_image_from_eigen_vectors(e_dir,
                                                                        shape))
    background_image_temp = background_image.copy()
    background_image[background_image < 0] = 0

    background_image = (((background_image - np.percentile(background_image, 1)) / (
            np.percentile(background_image, percent) - np.percentile(
        background_image, 1))))
    if np.percentile(background_image, 30) > .25:
        background_image = (
            ((background_image_temp - np.percentile(background_image_temp, 10)) / (
                    np.percentile(background_image_temp, percent) - np.percentile(
                background_image_temp, 10))))

    background_image[background_image > 1] = 1
    background_image = background_image * 255
    background_image[background_image < 0] = 0
    print(shape)

    combined_image = np.dstack([np.zeros(shape), background_image,
                                np.zeros(shape)])
    roi_image_combined = np.dstack(
        [np.zeros(shape), np.zeros(shape),
         np.zeros(shape)])
    roi_image_combined_single = np.dstack(
        [np.zeros(shape), np.zeros(shape),
         np.zeros(shape)])
    data_list = []
    colors = []
    if data_1 != "":
        data_list.append(data_1)
        colors.append(color_1)
    if data_2 != "":
        data_list.append(data_2)
        colors.append(color_2)
    if data_3 != "":
        data_list.append(data_3)
        colors.append(color_3)
    if data_4 != "":
        data_list.append(data_4)
        colors.append(color_4)
    if type(offset) != list:
        offset = [offset for _ in data_list]
    roi_image = create_roi_image(shape, colors[0], data_list[0], blobs=True, offset=0)
    # roi_image_single = create_roi_image(shape, (1, 0, 0), data_list[0], blobs=True,
    #                                     offset=offset[0])
    if blobs:
        roi_image_combined += roi_image
    else:
        roi_image_combined[roi_image != 0] = roi_image[roi_image != 0]
    roi_image_combined[roi_image_combined > 255] = 255
    # roi_image_combined_single += roi_image_single
    roi_image_sum = np.sum(roi_image_combined, axis=2)
    roi_image_combined_single_sum = np.sum(roi_image_combined_single, axis=2)

    # combined_image[roi_image_combined_single_sum != 0] = [0, 0, 0]
    combined_image += roi_image_combined
    # if blobs and len(data_list) == 2:
    #     combined_image[roi_image_combined_single_sum > 1] = overlap_c
    combined_image = combined_image.astype(int)
    # fig, ax = plt.subplots(1, 1)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5.12, 5.12)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # fig.set_size_inches(w, h)
    ax.imshow(combined_image.astype(np.uint8), vmin=0, vmax=255)
    for x, color, offset_f in list(zip(data_list, colors, offset))[1:]:
        roi_image = create_roi_image_cont(shape, color, x, ax, offset=offset_f)
        # roi_image_single = create_roi_image(shape, (1, 0, 0), x, blobs=blobs,
        #                                     offset=offset_f)
        # if blobs:
        #     roi_image_combined += roi_image
        # # else:
        #     # roi_image_combined[roi_image != 0] = roi_image[roi_image != 0]
        #
        # roi_image_combined_single += roi_image_singlecombined_single

    fig.savefig(out_file)
    # plt.show()


# TODO CROP IMAGE TO 235-235
if __name__ == '__main__':
    fire.Fire(create_graph)
