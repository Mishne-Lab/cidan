import logging
from functools import reduce

import numpy as np
from PIL import Image
from dask import delayed
from skimage import feature, measure
from skimage.measure import find_contours

from cidan.LSSC.SpatialBox import SpatialBox
from cidan.LSSC.functions.data_manipulation import pixel_num_to_2d_cord
from cidan.LSSC.functions.eigen import loadEigenVectors
from cidan.LSSC.functions.pickle_funcs import *
from cidan.LSSC.functions.roi_extraction import roi_extract_image, combine_rois
from cidan.LSSC.functions.roi_filter import filterRoiList
from cidan.LSSC.process_data import process_data

logger1 = logging.getLogger("cidan.DataHandler")


def calculate_roi_extraction(self, progress_signal=None):
    """
    Extracts Rois and sets them to self.rois
    """
    if self.global_params["need_recalc_eigen_params"] or self.global_params[
        "need_recalc_roi_extraction_params"] or self.global_params[
        "need_recalc_box_params"] or self.global_params[
        "need_recalc_dataset_params"] or \
            self.global_params["need_recalc_filter_params"]:
        assert (int(
            self.box_params[
                "total_num_spatial_boxes"] ** .5)) ** 2 == self.box_params[
                   "total_num_spatial_boxes"], "Please make sure Number of Spatial Boxes is a square number"
        try:
            self.calculate_filters(progress_signal=progress_signal)
            eigen_need_recalc = self.global_params["need_recalc_eigen_params"] or \
                                self.global_params[
                                    "need_recalc_box_params"]
            self.global_params["need_recalc_eigen_params"] = False
            self.global_params[
                "need_recalc_roi_extraction_params"] = False
            temp_params = self.box_params.copy()
            self.global_params[
                "need_recalc_box_params"] = False
            image_data = []
            for num, trial_num in enumerate(self._trials_loaded_indices):
                if type(self.dataset_trials_filtered_loaded[num]) == type(delayed(min)()):
                    image_data.append(self.compute_trial_filter_step(
                            trial_num,
                            loaded_num=num, save_data=False))
                else:
                    image_data.append(self.dataset_trials_filtered_loaded[num])
            self.rois = process_data( test_images=False, test_output_dir="",
                                     save_dir=self.save_dir_path,
                                      shape=self.shape,
                                     save_intermediate_steps=self.global_params[
                                         "save_intermediate_steps"],
                                     image_data_filtered=self.dataset_trials_filtered_loaded,
                                     image_data=self.dataset_list,
                                     crop= [self.dataset_params["crop_x"],self.dataset_params["crop_y"]] if self.dataset_params["crop_stack"] else False,
                                     slicing = [self.dataset_params["slice_start"][0],self.dataset_params["slice_every"][0]] if self.dataset_params["slice_stack"] else [0,1],

                                      eigen_vectors_already_generated=(not
                                                                      eigen_need_recalc) and
                                                                     self.global_params[
                                                                         "save_intermediate_steps"] and self.eigen_vectors_exist,
                                     save_embedding_images=True,
                                     total_num_time_steps=self.box_params[
                                         "total_num_time_steps"],
                                     total_num_spatial_boxes=self.box_params[
                                         "total_num_spatial_boxes"],
                                     spatial_overlap=self.box_params[
                                                         "spatial_overlap"] // 2,

                                     metric=self.eigen_params["metric"],
                                     knn=self.eigen_params["knn"],
                                     accuracy=self.eigen_params["accuracy"],
                                     eigen_accuracy=self.eigen_params[
                                         "eigen_accuracy"],
                                     connections=self.eigen_params[
                                         "connections"],
                                     normalize_w_k=self.eigen_params[
                                         "normalize_w_k"],
                                     num_eig=self.eigen_params["num_eig"],
                                     merge=self.roi_extraction_params["merge"],
                                     num_rois=self.roi_extraction_params[
                                         "num_rois"],
                                     refinement=self.roi_extraction_params[
                                         "refinement"],
                                     num_eigen_vector_select=
                                     self.roi_extraction_params[
                                         "num_eigen_vector_select"],
                                     max_iter=self.roi_extraction_params[
                                         "max_iter"],
                                     roi_size_min=self.roi_extraction_params[
                                         "roi_size_min"],
                                     fill_holes=self.roi_extraction_params[
                                         "fill_holes"],

                                     elbow_threshold_method=
                                     self.roi_extraction_params[
                                         "elbow_threshold_method"],
                                     elbow_threshold_value=
                                     self.roi_extraction_params[
                                         "elbow_threshold_value"],
                                     eigen_threshold_method=
                                     self.roi_extraction_params[
                                         "eigen_threshold_method"],
                                     eigen_threshold_value=
                                     self.roi_extraction_params[
                                         "eigen_threshold_value"],
                                     merge_temporal_coef=
                                     self.roi_extraction_params[
                                         "merge_temporal_coef"],
                                     roi_size_max=self.roi_extraction_params[
                                         "roi_size_max"],
                                     pca=self.filter_params["pca"],
                                     pca_data=self.pca_decomp if self.filter_params[
                                         "pca"] else False,
                                     roi_eccentricity_limit=
                                     self.roi_extraction_params[
                                         "roi_eccentricity_limit"],
                                     local_max_method=
                                     self.roi_extraction_params[
                                         "local_max_method"],

                                     progress_signal=progress_signal)
            self.box_params_processed = temp_params
            self.save_new_param_json()
            roi_valid_list = filterRoiList(self.rois, self.shape)
            self.rois = [x for x, y in zip(self.rois, roi_valid_list) if
                         y >= self.roi_extraction_params[
                             "roi_circ_threshold"]]
            self.save_rois(self.rois)
            print("Calculating Time Traces")
            self.gen_roi_display_variables()
            self.calculate_time_traces()

            self.rois_loaded = True
        except FileNotFoundError as e:
            self.global_params["need_recalc_eigen_params"] = False
            logger1.error(e)
            print(
                "Please try again there was an internal error in the roi extraction process")
            raise AssertionError()


def gen_roi_display_variables(self):
    """
    Generates the other variables associated with displaying ROIs
    """
    self.roi_circ_list = filterRoiList(self.rois, self.shape,
                                       self.roi_extraction_params[
                                           "roi_circ_threshold"])
    roi_list_2d_cord = [
        pixel_num_to_2d_cord(x, volume_shape=self.shape) for x in
        self.rois]
    self.roi_max_cord_list = [np.max(x, axis=1) for x in roi_list_2d_cord]
    self.roi_min_cord_list = [np.min(x, axis=1) for x in roi_list_2d_cord]
    self.pixel_with_rois_flat = np.zeros(
        [self.shape[0] * self.shape[1]])
    self.pixel_with_rois_color_flat = np.zeros(
        [self.shape[0] * self.shape[1], 3])
    edge_roi_image = np.zeros([self.shape[0], self.shape[1]])
    for num, roi in enumerate(self.rois):
        cur_color = self.color_list[num % len(self.color_list)]
        self.pixel_with_rois_flat[roi] = num + 1
        self.pixel_with_rois_color_flat[roi] = cur_color
        roi_edge = np.zeros(
            [self.shape[0] * self.shape[1]])
        roi_edge[roi] = 255
        edge_roi_image += feature.canny(np.reshape(roi_edge,
                                                   [self.shape[0],
                                                    self.shape[1]]))

    edge_roi_image[edge_roi_image > 255] = 255
    self.edge_roi_image_flat = np.reshape(edge_roi_image, [-1, 1]) * 255
    self.pixel_with_rois_color = np.reshape(self.pixel_with_rois_color_flat,
                                            [self.shape[0],
                                             self.shape[1], 3])
    # self.all_roi_contours = []
    # pixel_with_rois = self.pixel_with_rois_flat.reshape([self.shape[0],self.shape[1]])
    # for num in range(len(self.rois)):
    #
    #
    #     image_temp = pixel_with_rois==num+1
    #
    #     # edge = feature.canny(
    #     #     np.sum(image_temp, axis=2) / np.max(np.sum(image_temp, axis=2)))
    #     # image[edge] = 1
    #     # image_temp = ndimage.morphology.binary_dilation(image_temp)
    #     test = measure.label(image_temp, background=0, connectivity=1)
    #     # image_temp = ndimage.morphology.binary_erosion(image_temp)
    #     #
    #     # image_temp = ndimage.morphology.binary_erosion(image_temp)
    #     # image_temp = ndimage.binary_closing(image_temp)
    #     # print(test.max())
    #     for x in range(test.max()):
    #         image = np.zeros((self.shape[0], self.shape[1]), dtype=float)
    #         image[test == x + 1] = 1
    #         contour = find_contours(image, .3)
    #         self.all_roi_contours.append(contour)
    try:
        self.eigen_norm_image = np.asarray(Image.open(
            os.path.join(self.save_dir_path,
                         "embedding_norm_images/embedding_norm_image.png")))
    except:
        print("Can't generate eigen Norm image please try again")


def genRoiFromPoint(self, point, growth_factor=1.0):
    """
    Generates an roi from an initial point in the image
    Parameters
    ----------
    point : List[int, int]
        2d cordinates of a point in the image
    growth_factor : float
        how much to grow the roi

    Returns
    -------
    the pixels in said roi in there 1d form
    """
    spatial_boxes = [SpatialBox(box_num=x,
                                total_boxes=self.box_params_processed[
                                    "total_num_spatial_boxes"],
                                image_shape=self.shape,
                                spatial_overlap=self.box_params_processed[
                                                    "spatial_overlap"] // 2) for x
                     in
                     range(self.box_params_processed["total_num_spatial_boxes"])]
    boxes_point_in = [x for x in spatial_boxes if x.pointInBox(point)]

    rois = []
    for box in boxes_point_in:
        all_eigen_vectors_list = []
        for temporal_box_num in range(
                self.box_params_processed["total_num_time_steps"]):
            all_eigen_vectors_list.append(
                loadEigenVectors(spatial_box_num=box.box_num,
                                 time_box_num=temporal_box_num,
                                 save_dir=self.save_dir_path).compute())

        all_eigen_vectors = np.hstack(all_eigen_vectors_list)
        box_point = box.point_to_box_point(point)
        box_point_1d = box_point[0] * box.shape[1] + box_point[1]
        roi = roi_extract_image(e_vectors=all_eigen_vectors,
                                original_shape=box.shape, original_2d_vol=None,
                                merge=False, box_num=box.box_num,
                                num_rois=1, refinement=True,
                                num_eigen_vector_select=
                                self.roi_extraction_params[
                                    "num_eigen_vector_select"],
                                max_iter=1,
                                roi_size_min=0,
                                fill_holes=self.roi_extraction_params[
                                    "fill_holes"],

                                elbow_threshold_method=True,
                                elbow_threshold_value=growth_factor,
                                eigen_threshold_method=
                                self.roi_extraction_params[
                                    "eigen_threshold_method"],
                                eigen_threshold_value=
                                self.roi_extraction_params[
                                    "eigen_threshold_value"],
                                merge_temporal_coef=
                                self.roi_extraction_params[
                                    "merge_temporal_coef"],
                                roi_size_limit=1000,
                                initial_pixel=box_point_1d,
                                print_info=False,
                                roi_eccentricity_limit=1).compute()

        if len(roi) > 0 and box_point_1d in roi[0]:
            rois.append(box.redefine_spatial_cord_1d(roi).compute()[0])
    if len(rois) > 0:
        final_roi = reduce(combine_rois, rois)
    else:
        return []

    return final_roi
