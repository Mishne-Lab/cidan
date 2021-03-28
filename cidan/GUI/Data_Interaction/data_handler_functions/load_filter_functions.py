import logging

import dask
import numpy as np
import zarr
from dask import delayed
from tifffile import tifffile

from cidan.LSSC.functions.data_manipulation import load_tif_stack, filter_stack, \
    applyPCA, load_tif_stack_trial, slice_crop_stack
from cidan.LSSC.functions.pickle_funcs import *

logger1 = logging.getLogger("cidan.DataHandler")


def reset_data(self):
    self.update_trial_list()
    self.dataset_trials_filtered = [False] * len(self.trials_all)
    for num, trial_num in enumerate(self._trials_loaded_indices):
        self.dataset_trials_filtered[
                            trial_num] = self.compute_trial_filter_step(
                            trial_num,
                            loaded_num=num)
    if self.filter_params["pca"]:
        self.pca_decomp = [False] * len(self.trials_loaded_time_trace_indices)
    # for trial_num in self._trials_loaded_indices:
        # self.dataset_trials_filtered[trial_num] = zarr.open(
        #     os.path.join(self.save_dir_path,
        #                  'temp_files/%s.zarr' % self.trials_all[trial_num]),
        #     mode="r")
        # if self.filter_params["pca"]:
        #     self.pca_decomp[trial_num] = zarr.open(
        #         os.path.join(self.save_dir_path,
        #                      'temp_files/%s_pca.zarr' % self.trials_all[trial_num]),
        #         mode="r")
    # max_image_zarr = zarr.open(
    #     os.path.join(self.save_dir_path,
    #                  'temp_files/max.zarr'),
    #     mode="r")
    self.max_images = [False] * len(self.trials_loaded_time_trace_indices)
    # mean_image_zarr = zarr.open(
    #     os.path.join(self.save_dir_path,
    #                  'temp_files/mean.zarr'),
    #     mode="r")
    self.mean_images = [False] * len(self.trials_loaded_time_trace_indices)
    # for x in range(len(self._trials_loaded_indices)):
    #     self.max_images[x] = max_image_zarr[:, :, x]
    #     self.mean_images[x] = mean_image_zarr[:, :, x]
    # self.shape = [
    #     self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[1],
    #     self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[2]]
    # if self.dataset_params["crop_x"][1] == 0:
    #     self.dataset_params["crop_x"][1] = self.shape[0]
    #     self.dataset_params["crop_y"][1] = self.shape[1] # TODO make sure to re-add this in somewhere
    # self.mean_images = [np.mean(x[:], axis=0) for x in
    #                     self.dataset_trials_filtered_loaded]
    #
    # self.max_images = [np.max(x[:], axis=0) for x in
    #                    self.dataset_trials_filtered_loaded]
    self.global_params["need_recalc_filter_params"] = False
    self.global_params["need_recalc_dataset_params"] = False


def transform_data_to_zarr(self):
    # used in large data mode to turn a single tiff file into a dataset where it is easy to extract only the time frams that are necessary
    image = tifffile.imread(os.path.join(self.dataset_params["dataset_folder_path"],
                                         self.dataset_params[
                                             "original_folder_trial_split"]))
    z1 = zarr.open(os.path.join(self.save_dir_path,
                                'temp_files/dataset.zarr'), mode='w',
                   shape=image.shape,
                   chunks=(4, image.shape[1],image.shape[2]), dtype=image.dtype)
    z1[:] = image


def load_dataset(self, path_list):
    # TODO add in zarr load support to here
    if self.load_into_mem:
        if self.single_dataset_mode:

            self.dataset_list = [load_tif_stack(path_list[0], convert_to_32=False)]
        else:
            self.dataset_list = dask.compute(
                *[delayed(load_tif_stack)(x) for x in path_list])
        self.total_size = [self.dataset_list[0].shape[1], self.dataset_list[0].shape[2]]
    else:
        if self.single_dataset_mode:
            self.dataset_list= [load_tif_stack(path_list[0], convert_to_32=False)]
        else:
            self.dataset_list = dask.compute(
                *[delayed(load_tif_stack)(x) for x in path_list])
        self.total_size = [self.dataset_list[0].shape[1],self.dataset_list[0].shape[2]]


def calculate_dataset(self) -> np.ndarray:
    """
    Loads each trial, applying crop and slicing, sets them to self.dataset_trials

    """
    # TODO make it so it does't load the dataset every time

    dataset_trials = [False] * len(self.trials_all)
    for trial_num in self._trials_loaded_indices:
        dataset_trials[trial_num] = self.load_trial_dataset_step(trial_num)
    dataset_trials = dask.compute(*dataset_trials)
    self.global_params["need_recalc_dataset_params"] = False
    self.shape = [dataset_trials[self._trials_loaded_indices[0]].shape[1],
                  dataset_trials[self._trials_loaded_indices[0]].shape[2]]
    print(dataset_trials[0])
    if self.dataset_params["crop_x"][1] == 0:
        self.dataset_params["crop_x"][1] = self.shape[0]
        self.dataset_params["crop_y"][1] = self.shape[1]
    print("Finished Calculating Dataset")
    return dataset_trials

@delayed
def load_dataset_trial(self, trial_num):
    """
    Delayed function step that loads a single trial

    should only be used in large data mode
    Parameters
    ----------
    trial_num : int
        trial num to load in trials_all

    Returns
    -------
    Trial as a np.ndarray
    """
    # TODO make these params correct
    total_size, stack = load_tif_stack_trial(
        path=os.path.join(self.dataset_params["dataset_folder_path"],
                          self.trials_all[trial_num] if not self.dataset_params[
                              "trial_split"] else self.dataset_params[
                              "original_folder_trial_split"]),
        # filter=False,
        # median_filter=False,
        # median_filter_size=(1, 3, 3),
        # z_score=False,
        # slice_stack=self.dataset_params[
        #     "slice_stack"],
        # slice_start=self.dataset_params[
        #     "slice_start"],
        #
        # slice_every=self.dataset_params[
        #     "slice_every"],
        # crop_stack=self.dataset_params[
        #     "crop_stack"],
        # crop_x=self.dataset_params[
        #     "crop_x"],
        # crop_y=self.dataset_params[
        #     "crop_y"], load_into_mem=self.load_into_mem,
        trial_split=self.dataset_params["trial_split"],
        trial_split_length=self.dataset_params["trial_length"], trial_num=trial_num,
        # zarr_path=False if not self.dataset_params["single_file_mode"] else
        # os.path.join(self.save_dir_path, "temp_files/dataset.zarr")
    )
    self.total_size = total_size
    return stack

@delayed
def compute_trial_filter_step(self, trial_num, loaded_num,dataset=False, save_data=True ):
    """
    Delayed function step that applies filter to a single trial
    Parameters
    ----------
    trial_num : int
        trial num to load in trials_all
    dataset : ndarray
        Dataset to process if false load dataset
    loaded_num :
        trial num in array

    Returns
    -------
    Filtered trial as a np.ndarray
    """
    # NEED to add the data spliting code to here

    if self.dataset_params[
        "trial_split"]:
        cur_stack = self.dataset_list[0]
        cur_stack = cur_stack[trial_num * self.dataset_params[
        "trial_length"]:(loaded_num + 1) * self.dataset_params[
        "trial_length"],
        :, :]
    else:
        cur_stack = self.dataset_list[trial_num][:]
    cur_stack =cur_stack.astype(np.float32)

    cur_stack = slice_crop_stack(cur_stack,slice_stack=self.dataset_params[
            "slice_stack"],
        slice_start=self.dataset_params[
            "slice_start"],

        slice_every=self.dataset_params[
            "slice_every"],
        crop_stack=self.dataset_params[
            "crop_stack"],
        crop_x=self.dataset_params[
            "crop_x"],
        crop_y=self.dataset_params[
            "crop_y"])

    cur_stack = filter_stack(

        stack=cur_stack,
        median_filter_size=(self.filter_params[
                                "median_filter_size"],
                            self.filter_params[
                                "median_filter_size"],
                            self.filter_params[
                                "median_filter_size"]),
        median_filter=self.filter_params[
            "median_filter"],
        z_score=self.filter_params["z_score"],
        hist_eq=self.filter_params["hist_eq"],
        localSpatialDenoising=self.filter_params["localSpatialDenoising"])
    self.shape = [cur_stack.shape[1], cur_stack.shape[2]]

    if not self.load_into_mem:

        z1 = zarr.open(os.path.join(self.save_dir_path,
                                    'temp_files/%s.zarr' % self.trials_all[
                                        trial_num]), mode='w',
                       shape=cur_stack.shape,
                       chunks=(32, 128, 128), dtype=np.float32)
        z1[:] = cur_stack

            # self.temporal_correlation_images[
            #     loaded_num] = calculate_temporal_correlation(cur_stack).compute()




    self.mean_images[loaded_num] = np.mean(cur_stack, axis=0)
    self.max_images[loaded_num] = np.max(cur_stack, axis=0)
            # self.temporal_correlation_images[
            #     loaded_num] = calculate_temporal_correlation(
            #     cur_stack).compute()
    if self.filter_params["pca"]:
        pca = applyPCA(cur_stack, self.filter_params["pca_threshold"])
        if self.load_into_mem:
            self.pca_decomp[loaded_num] = pca
        else:
            z2 = zarr.open(os.path.join(self.save_dir_path,
                                        'temp_files/%s_pca.zarr' % self.trials_all[
                                            trial_num]), mode='w',
                           shape=pca.shape,
                           chunks=(3, 64, 64))
            z2[:] = pca
            self.pca_decomp[loaded_num] = z2

    # with open(os.path.join(self.save_dir_path,
    #                        'temp_files/filter/%s' % self.trials_all[
    #                            trial_num]), "w") as f:
    #     f.write("done")
    # printProgressBarFilter(
    #     total_num_spatial_boxes=self.box_params["total_num_spatial_boxes"],
    #     total_num_time_steps=self.box_params["total_num_time_steps"],
    #     save_dir=self.save_dir_path, progress_signal=self.progress_signal)

    if self.load_into_mem:
        if save_data:
            self.dataset_trials_filtered[trial_num] = cur_stack
        return cur_stack
    else:
        if save_data:
            self.dataset_trials_filtered[trial_num] = z1
        return z1


# def calculate_filters(self, progress_signal=None, auto_crop=False):
#     """
#     Applies filter to each trial, sets them to self.dataset_trials_filtered
#
#     Returns
#     -------
#     A list of filtered trials
#     """
#     if not self.auto_crop:
#         self.dataset_params["auto_crop"] = auto_crop
#     if self.global_params["need_recalc_filter_params"] or self.global_params[
#         "need_recalc_dataset_params"] or \
#             not hasattr(self, "dataset_trials_filtered"):
#         self.progress_signal = progress_signal
#         print("Started Calculating Filters")
#         reset_data(self)
#
#         save_dir = self.save_dir_path
#         if not os.path.isdir(os.path.join(save_dir, "temp_files/filter")):
#             os.mkdir(os.path.join(save_dir, "temp_files/filter"))
#         filelist = [f for f in
#                     os.listdir(os.path.join(save_dir, "temp_files/filter"))]
#         for f in filelist:
#             os.remove(
#                 os.path.join(os.path.join(save_dir, "temp_files/filter"), f))
#
#         printProgressBarFilter(
#             total_num_spatial_boxes=self.box_params["total_num_spatial_boxes"],
#             total_num_time_steps=self.box_params["total_num_time_steps"],
#             save_dir=self.save_dir_path, progress_signal=self.progress_signal)
#         if self.auto_crop:
#             self.suggested_crops = [[[0, 0], [0, 0]] for _ in self.trials_all]
#
#
#         self.temporal_correlation_images = [False] * len(
#             self._trials_loaded_indices)
#         if not self.load_into_mem:
#             for num, trial_num in enumerate(self._trials_loaded_indices):
#                 self.dataset_trials_filtered[
#                     trial_num] = self.load_trial_filter_step(
#                     trial_num, self.load_trial_dataset_step(trial_num),
#                     loaded_num=num)
#                 if num % 3 == 0:
#                     self.dataset_trials_filtered = list(
#                         dask.compute(*self.dataset_trials_filtered))
#
#             self.dataset_trials_filtered = list(
#                 dask.compute(*self.dataset_trials_filtered))
#         else:
#             for num, trial_num in enumerate(self._trials_loaded_indices):
#                 self.dataset_trials_filtered[
#                     trial_num] = self.load_trial_filter_step(
#                     trial_num, self.load_trial_dataset_step(trial_num),
#                     loaded_num=num)
#             if num % 7 == 0:
#                 self.dataset_trials_filtered = list(
#                     dask.compute(*self.dataset_trials_filtered))
#             self.dataset_trials_filtered = list(
#                 dask.compute(*self.dataset_trials_filtered))
#         self.shape = [
#             self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[1],
#             self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[2]]
#         if self.dataset_params["crop_x"][1] == 0:
#             self.dataset_params["crop_x"][1] = self.shape[0]
#             self.dataset_params["crop_y"][1] = self.shape[1]
#         if self.auto_crop:
#             crop = [[max([x[0][0] for x in self.suggested_crops]),
#                      -max([x[0][1] for x in self.suggested_crops])],
#                     [max([x[1][0] for x in self.suggested_crops]),
#                      -max([x[1][1] for x in self.suggested_crops])]]
#
#             self.dataset_params["crop_x"][0] = 0 + \
#                                                crop[0][0]
#             self.dataset_params["crop_x"][1] = self.total_size[0] + \
#                                                crop[0][1]
#             self.dataset_params["crop_y"][0] = 0 + \
#                                                crop[1][0]
#             self.dataset_params["crop_y"][1] = self.total_size[1] + \
#                                                crop[1][1]
#             if crop[0][1] == 0:
#                 crop[0][1] = self.total_size[0]
#             if crop[1][1] == 0:
#                 crop[1][1] = self.total_size[1]
#             for trial_num in self._trials_loaded_indices:
#                 self.dataset_trials_filtered[trial_num] = \
#                     self.dataset_trials_filtered[trial_num][
#                     :, crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
#                 self.mean_images[trial_num] = self.mean_images[trial_num][
#                                               crop[0][0]:crop[0][1],
#                                               crop[1][0]:crop[1][1]]
#                 self.max_images[trial_num] = self.max_images[trial_num][
#                                              crop[0][0]:crop[0][1],
#                                              crop[1][0]:crop[1][1]]
#                 # self.temporal_correlation_images[trial_num] = \
#                 #     self.temporal_correlation_images[trial_num][
#                 #     crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
#                 if self.filter_params["pca"]:
#                     self.pca_decomp[trial_num] = self.pca_decomp[trial_num][
#                                                  :, crop[0][0]:crop[0][1],
#                                                  crop[1][0]:crop[1][1]]
#             self.dataset_params["crop_stack"] = True
#             self.dataset_params["auto_crop"] = False
#             self.shape = [
#                 self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[
#                     1],
#                 self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[
#                     2]]
#
#         mean_image_stack = np.dstack(self.mean_images)
#         with open(os.path.join(self.save_dir_path,
#                                'temp_files/filter/mean'), "w") as f:
#             f.write("done")
#         printProgressBarFilter(
#             total_num_spatial_boxes=self.box_params["total_num_spatial_boxes"],
#             total_num_time_steps=self.box_params["total_num_time_steps"],
#             save_dir=self.save_dir_path, progress_signal=self.progress_signal)
#
#         max_image_stack = np.dstack(self.max_images)
#         with open(os.path.join(self.save_dir_path,
#                                'temp_files/filter/max'), "w") as f:
#             f.write("done")
#         mean_image_zarr = zarr.open(os.path.join(self.save_dir_path,
#                                                  'temp_files/mean.zarr'), mode='w',
#                                     shape=mean_image_stack.shape,
#                                     chunks=(40, 40, 1))
#         mean_image_zarr[:] = mean_image_stack
#
#         max_image_zarr = zarr.open(os.path.join(self.save_dir_path,
#                                                 'temp_files/max.zarr'), mode='w',
#                                    shape=mean_image_stack.shape,
#                                    chunks=(40, 40, 1))
#         max_image_zarr[:] = max_image_stack
#         printProgressBarFilter(
#             total_num_spatial_boxes=self.box_params["total_num_spatial_boxes"],
#             total_num_time_steps=self.box_params["total_num_time_steps"],
#             save_dir=self.save_dir_path, progress_signal=self.progress_signal)
#
#         # self.mean_images = [np.mean(x[:], axis=0) for x in
#         #                     self.dataset_trials_filtered_loaded]
#         #
#         # self.max_images = [np.max(x[:], axis=0) for x in
#         #                    self.dataset_trials_filtered_loaded]
#
#         # self.temporal_correlation_image = calculate_temporal_correlation(self.dataset_filtered)
#         self.global_params["need_recalc_filter_params"] = False
#         self.global_params["need_recalc_dataset_params"] = False
#
#         # self.global_params["need_recalc_box_params"] = True
#         self.save_new_param_json()
#         self.delete_roi_vars()
#         # self.global_params["need_recalc_eigen_params"] = True
#         if self.filter_params["pca"]:
#             with open(os.path.join(self.save_dir_path,
#                                    'pca_shape.text'), "w") as f:
#                 f.write("shapes")
#                 f.write(str([x.shape for x in self.pca_decomp if type(x) != bool]))
#     return self.dataset_trials_filtered
