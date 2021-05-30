import json
import logging
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from skimage import measure
from skimage.measure import find_contours

from cidan.LSSC.SpatialBox import SpatialBox
from cidan.LSSC.functions.data_manipulation import reshape_to_2d_over_time
from cidan.LSSC.functions.pickle_funcs import *
from cidan.LSSC.functions.spatial_footprint import classify_components_ep

logger1 = logging.getLogger("cidan.DataHandler")


def calculate_statistics(self):
    statistics = [False for x in range(len(self.rois))]
    A = np.zeros([self.edge_roi_image_flat.shape[0], len(self.rois)], dtype=int)
    for num, roi in enumerate(self.rois):
        A[roi, num] = 1
    rval, significant_samples = classify_components_ep(
        self.dataset_trials_filtered_loaded, A,
        np.vstack([self.get_time_trace(x + 1) for x in range(len(self.rois))]))
    pass


def export(self, matlab, background_images, color_maps):
    # temp_type = self.time_trace_params["time_trace_type"]
    # for time_type in ["Mean", "DeltaF Over F"]:
    #     for denoise in [True, False]:
    #         self.time_trace_params["denoise"] = denoise
    #         self.time_trace_params["time_trace_type"] = time_type
    #         self.calculate_time_traces()
    #         # if time_type == "Mean" and denoise:
    #         #     self.calculate_statistics()
    # self.time_trace_params["time_trace_type"] = temp_type
    self.save_rois(self.rois)
    roi_save_object = []
    spatial_box = SpatialBox(0, 1, image_shape=self.shape, spatial_overlap=0)
    for num, roi in enumerate(self.rois):
        if self.dataset_params["crop_stack"]:

            cords = [[x[0] + self.dataset_params["crop_x"][0],
                      x[1] + self.dataset_params["crop_y"][0]] for x in
                     spatial_box.convert_1d_to_2d(roi)]
        else:
            cords = spatial_box.convert_1d_to_2d(roi)
        curr_roi = {"id": num, "coordinates": cords}
        roi_save_object.append(curr_roi)

    with open(os.path.join(self.save_dir_path, "roi_list.json"), "w") as f:
        json.dump(roi_save_object, f)
    if matlab:
        test = {x[:31].replace(" ", "_"): np.vstack(self.time_traces[x]) for x in
                self.time_traces.keys()}

        savemat(os.path.join(self.save_dir_path, "time_traces.mat"), {"data": test},
                appendmat=True)
        savemat(os.path.join(self.save_dir_path, "rois.mat"), {"data": self.rois},
                appendmat=True)

    shape = self.shape

    if not os.path.isdir(os.path.join(self.save_dir_path, "images/")):
        os.mkdir(os.path.join(self.save_dir_path, "images/"))
    rois = []
    for num, roi in enumerate(self.rois):
        cords = spatial_box.convert_1d_to_2d(roi)
        rois.append(cords)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(shape[1] / 100, shape[0] / 100)

    for background_image_name in background_images:
        if background_image_name == "mean":
            background_image = np.mean(
                [self.mean_images[x] for x in self._trials_loaded_indices], axis=0)
        elif background_image_name == "max":
            background_image = np.max(
                [self.max_images[x] for x in self._trials_loaded_indices], axis=0)
        elif background_image_name == "blank":
            background_image = np.zeros(shape)
        else:  # eigen norm is default
            background_image = create_image_from_eigen_vectors(
                os.path.join(self.save_dir_path, "eigen_vectors/"), shape)
        if not background_image_name == "zeros":
            background_image = scale_background(background_image)

        for color_map_name in color_maps:
            # first the blob plot
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if color_map_name == "green":
                ax.imshow(np.dstack(
                    [np.zeros(shape), background_image, np.zeros(shape)]).astype(
                    np.uint8), vmin=0, vmax=255)

            else:
                ax.imshow(background_image.astype(np.uint8), vmin=0, vmax=255,
                          cmap=color_map_name)
            ax.imshow(create_roi_image_blob(color_list=self.color_list, rois=rois,
                                            shape=shape).astype(np.uint8), vmin=0,
                      vmax=255)
            fig.savefig(os.path.join(self.save_dir_path, "images/",
                                     "roi_blob_%s_image_%s.png" % (
                                     background_image_name, color_map_name)))
            fig.clf()
            # now the outline plot
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if color_map_name == "green":
                ax.imshow(np.dstack(
                    [np.zeros(shape), background_image, np.zeros(shape)]).astype(
                    np.uint8), vmin=0, vmax=255)

            else:
                ax.imshow(background_image.astype(np.uint8), vmin=0, vmax=255,
                          cmap=color_map_name)
            plot_roi_image_contour(shape, (1, 1, 1), rois, ax)
            fig.savefig(os.path.join(self.save_dir_path, "images/",
                                     "roi_outline_%s_image_%s.png" % (
                                     background_image_name, color_map_name)))
            fig.clf()
            # now just background plot
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if color_map_name == "green":
                ax.imshow(np.dstack(
                    [np.zeros(shape), background_image, np.zeros(shape)]).astype(
                    np.uint8), vmin=0, vmax=255)

            else:
                ax.imshow(background_image.astype(np.uint8), vmin=0, vmax=255,
                          cmap=color_map_name)
            fig.savefig(os.path.join(self.save_dir_path, "images/",
                                     "background_%s_image_%s.png" % (
                                     background_image_name, color_map_name)))
            fig.clf()


def save_image(self, image, path):
    plt.imsave(path, image.reshape((self.shape[0], self.shape[1], 3)).astype(np.uint8),
               vmin=0, vmax=255)


def create_image_from_eigen_vectors(path, shape, vectors=None):
    if vectors is None:
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        vectors = []
        for x in onlyfiles:
            with open(os.path.join(path, x), "rb") as file:
                vectors.append(pickle.load(file)[:, 1:])
        all_vectors = np.hstack(vectors)
    else:
        all_vectors = vectors
    all_vectors_sum = np.power(np.sum(np.power(all_vectors, 2), axis=1), .5)
    all_vectors_shaped = np.reshape(all_vectors_sum, shape)
    all_vectors_shaped[all_vectors_shaped < 0] = 0
    # if all_vectors_shaped.min()<0:
    #     all_vectors_shaped+=all_vectors_shaped.min()*-1
    return all_vectors_shaped * 255 / (all_vectors_shaped.max())


def scale_background(background_image):
    background_image[background_image < 0] = 0

    background_image = (((background_image - np.percentile(background_image, 1)) / (
            np.percentile(background_image, 97) - np.percentile(
        background_image, 1))))

    background_image[background_image > 1] = 1
    background_image = background_image * 255
    background_image[background_image < 0] = 0
    return background_image


def create_roi_image_blob(color_list, rois, shape):
    combined_image = np.dstack([np.zeros(shape), np.zeros(shape),
                                np.zeros(shape)])
    transparency = np.zeros(shape)

    for num, cords in enumerate(rois):
        for pixel in cords:
            combined_image[pixel[0], pixel[1]] = color_list[num % len(color_list)]
            transparency[pixel[0], pixel[1]] = 255

    combined_image = np.dstack([combined_image, transparency])
    return combined_image


def plot_roi_image_contour(shape, color, rois, fig):
    for num, cords in enumerate(rois):

        image_temp = np.zeros((shape[0], shape[1]), dtype=float)

        for pixel in cords:
            image_temp[pixel[0], pixel[1]] = 1

        # edge = feature.canny(
        #     np.sum(image_temp, axis=2) / np.max(np.sum(image_temp, axis=2)))
        # image[edge] = 1
        # image_temp = ndimage.morphology.binary_dilation(image_temp)
        test = measure.label(image_temp, background=0, connectivity=1)
        # image_temp = ndimage.morphology.binary_erosion(image_temp)
        #
        # image_temp = ndimage.morphology.binary_erosion(image_temp)
        # image_temp = ndimage.binary_closing(image_temp)
        # print(test.max())
        for x in range(test.max()):
            image = np.zeros((shape[0], shape[1]), dtype=float)
            image[test == x + 1] = 1
            contour = find_contours(image, .3)
            if len(contour) != 0:
                fig.plot(contour[0][:, 1], contour[0][:, 0], color=color,
                         linewidth=2)
            # plt.imshow(image)
            # fig.plot(contour[0][:, 1], contour[0][:, 0], color=color, linewidth=2)


def export_overlapping_rois(self, current_roi_num_1, current_roi_num_2):
    """THis is only used in dev version it allows the user to export time trace of all pixels in this roi and in overlapping rois"""
    current_roi_num_1 = current_roi_num_1 - 1
    current_roi_num_2 = current_roi_num_2 - 1
    pixels_in_1_not_2 = [x for x in self.rois[current_roi_num_1] if
                         x not in self.rois[current_roi_num_2]]
    pixels_in_2_not_1 = [x for x in self.rois[current_roi_num_2] if
                         x not in self.rois[current_roi_num_1]]
    pixels_in_1_and_2 = [x for x in self.rois[current_roi_num_2] if
                         x in self.rois[current_roi_num_1]]
    time_traces_1_not_2 = []
    time_traces_2_not_1 = []
    time_traces_1_and_2 = []

    for trial_num in self.trials_loaded_time_trace_indices:
        data = self.load_trial_dataset_step(trial_num).compute()
        data_2d = reshape_to_2d_over_time(data[:])
        del data
        time_traces_1_and_2.append(data_2d[pixels_in_1_and_2])
        time_traces_1_not_2.append(data_2d[pixels_in_1_not_2])
        time_traces_2_not_1.append(data_2d[pixels_in_2_not_1])
    time_traces_2_not_1 = np.hstack(time_traces_2_not_1)
    time_traces_1_not_2 = np.hstack(time_traces_1_not_2)
    time_traces_1_and_2 = np.hstack(time_traces_1_and_2)
    out = {"time_trace_2_not_1": time_traces_2_not_1,
           "time_trace_1_not_2": time_traces_1_not_2,
           "time_traces_1_and_2": time_traces_1_and_2}
    pickle_save(out, "roi_overlap.pickle", output_directory=self.save_dir_path)
