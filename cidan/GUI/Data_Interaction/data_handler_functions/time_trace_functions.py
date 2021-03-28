import logging

import dask
import numpy as np

from cidan.LSSC.functions.data_manipulation import reshape_to_2d_over_time
from cidan.LSSC.functions.pickle_funcs import *
from cidan.LSSC.functions.progress_bar import printProgressBar
from cidan.TimeTrace.neuropil import calculate_neuropil
from cidan.TimeTrace.waveletDenoise import waveletDenoise

logger1 = logging.getLogger("cidan.DataHandler")


def calculate_time_traces(self, report_progress=None):
    """
    Calculates the time traces for every roi in self.rois
    """
    self.neuropil_pixels = calculate_neuropil(image_shape=self.shape,
                                              roi_list=self.rois,
                                              roi_mask_flat=self.pixel_with_rois_flat,
                                              min_pixels=100)  # self.time_trace_params["min_neuropil_pixels"])
    self.neuropil_image_display = np.zeros(
        [self.shape[0] * self.shape[1]])
    for neurolpil in self.neuropil_pixels:
        self.neuropil_image_display[neurolpil] = 255
    self.neuropil_image_display = np.reshape(
        np.dstack([self.neuropil_image_display, np.zeros(
            [self.shape[0] * self.shape[1]]),
                   (self.pixel_with_rois_flat > 0) * 255]),
        [self.shape[0] * self.shape[1], 3])
    self.roi_time_trace_need_update = []
    for _ in range(len(self.rois)):
        self.roi_time_trace_need_update.append(False)
    self.time_traces = {x: [] for x in
                        list(self.time_trace_possibilities_functions)}
    for x in self.time_traces.keys():
        for _ in range(len(self.rois)):
            self.time_traces[x].append([])
            for _ in range(1 if self.single_dataset_mode else len(self.trials_all)):
                self.time_traces[x][-1].append(False)
    calc_list = []
    roi_time_traces_by_pixel = []
    for _ in range(len(self.rois)):
        roi_time_traces_by_pixel.append([])
        for _ in range(1 if self.single_dataset_mode else len(self.trials_all)):
            roi_time_traces_by_pixel[-1].append(False)
    roi_neuropil_traces_by_pixel = []
    for _ in range(len(self.rois)):
        roi_neuropil_traces_by_pixel.append([])
        for _ in range(1 if self.single_dataset_mode else len(self.trials_all)):
            roi_neuropil_traces_by_pixel[-1].append(False)


    if not self.real_trials:
        calc_list = []
        data_2d = reshape_to_2d_over_time(self.dataset_list[0][:])
        for roi in range(len(self.rois)):
            roi_time_traces_by_pixel[roi][0] = data_2d[self.rois[roi]]
            roi_neuropil_traces_by_pixel[roi][0] = data_2d[
                self.neuropil_pixels[roi]]
        if report_progress is not None:
            printProgressBar(self.trials_loaded_time_trace_indices[-1],
                             total=len(
                                 self.trials_loaded_time_trace_indices) + len(
                                 self.rois) + 2,
                             prefix="Time Trace Calculation Progress:",
                             suffix="Complete", progress_signal=report_progress)
        roi_time_traces_by_pixel_denoised = dask.compute(*[dask.delayed(waveletDenoise)(x[0]) for x in roi_time_traces_by_pixel])
        roi_neuropil_traces_by_pixel_denoised = dask.compute(*[dask.delayed(waveletDenoise)(x[0]) for x in roi_time_traces_by_pixel])

        for roi_counter, roi_data, neuropil_data,roi_data_denoised_combined,neuropil_data_denoised_combined in zip(range(len(self.rois)),
                                                        roi_time_traces_by_pixel,
                                                        roi_neuropil_traces_by_pixel,roi_time_traces_by_pixel_denoised,roi_neuropil_traces_by_pixel_denoised):
            roi_data_combined = roi_data[0]
            neuropil_data_combined = neuropil_data[0] #np.vstack([x[0] for x in roi_time_traces_by_pixel])

            for key in self.time_traces.keys():
                if "Denoised" in key:
                    self.time_traces[key][roi_counter] = [
                        self.time_trace_possibilities_functions[key](
                            roi_data_denoised_combined,
                            neuropil_data_denoised_combined,
                        )]
                else:
                    self.time_traces[key][roi_counter] = [
                        self.time_trace_possibilities_functions[key](
                            roi_data_combined, neuropil_data_combined)]
            if report_progress is not None:
                printProgressBar(
                    len(
                        self.trials_loaded_time_trace_indices) + roi_counter + 1,
                    total=len(
                        self.trials_loaded_time_trace_indices) + len(self.rois) + 2,
                    prefix="Time Trace Calculation Progress:",
                    suffix="Complete", progress_signal=report_progress)
    if self.real_trials:
        for trial_num in self.trials_loaded_time_trace_indices:
            data =self.dataset_list[trial_num]
            data_2d = reshape_to_2d_over_time(data[:])
            del data
            # if type(self.dataset_trials_filtered[trial_num]) == bool:
            #     data = self.load_trial_filter_step(
            #         trial_num).compute()
            #     self.dataset_trials_filtered[trial_num] = data
            #
            #     del data
            # else:
            #     data_2d = reshape_to_2d_over_time(
            #         self.dataset_trials_filtered[trial_num])[:]
            calc_list = []
            for roi in range(len(self.rois)):
                roi_time_traces_by_pixel[roi][trial_num] = data_2d[self.rois[roi]]
                roi_neuropil_traces_by_pixel[roi][trial_num] = data_2d[
                    self.neuropil_pixels[roi]]
            if report_progress is not None:
                printProgressBar(self.trials_loaded_time_trace_indices.index(trial_num),
                                 total=len(
                                     self.trials_loaded_time_trace_indices) + len(
                                     self.rois) + 2,
                                 prefix="Time Trace Calculation Progress:",
                                 suffix="Complete", progress_signal=report_progress)
        for roi_counter, roi_data, neuropil_data in zip(range(len(self.rois)),
                                                        roi_time_traces_by_pixel,
                                                        roi_neuropil_traces_by_pixel):
            roi_data_denoised = [waveletDenoise(x) if type(x) != bool else False for
                                 x in roi_data]
            neuropil_data_denoised = [
                waveletDenoise(x) if type(x) != bool else False for
                x in neuropil_data]
            for key in self.time_traces.keys():
                if "Denoised" in key:
                    for trial_num in self.trials_loaded_time_trace_indices:
                        self.time_traces[key][roi_counter][trial_num] = \
                            self.time_trace_possibilities_functions[key](
                                roi_data_denoised[trial_num],
                                neuropil_data_denoised[trial_num])
                else:
                    for trial_num in self.trials_loaded_time_trace_indices:
                        self.time_traces[key][roi_counter][trial_num] = \
                            self.time_trace_possibilities_functions[key](
                                roi_data[trial_num], neuropil_data[trial_num])
            if report_progress is not None:
                printProgressBar(
                    len(
                        self.trials_loaded_time_trace_indices) + roi_counter + 1,
                    total=len(
                        self.trials_loaded_time_trace_indices) + len(self.rois) + 2,
                    prefix="Time Trace Calculation Progress:",
                    suffix="Complete", progress_signal=report_progress)
    self.roi_time_trace_need_update = [False for _ in
                                       self.roi_time_trace_need_update]

    if os.path.isdir(self.save_dir_path):
        pickle_save(self.time_traces,
                    "time_traces",
                    output_directory=self.save_dir_path)
    self.rois_loaded = True


# def calculate_time_trace(self, roi_num, trial_num=None, data_2d=None):
#     """
#     Calculates a time trace for a certain ROI and save to time trace list
#     Parameters
#     ----------
#     roi_num
#         roi to calculate for this starts at [1..number of rois]
#     trial_num
#         indice of trial in trials_all starts at [0, number of trials-1] if
#         none then calculates for all trials
#     """
#     trial_nums = [trial_num]
#     if trial_num == None:
#         trial_nums = self.trials_loaded_time_trace_indices
#     for trial_num in trial_nums:
#         roi = self.rois[roi_num - 1]
#
#         if type(self.dataset_trials_filtered[
#                     trial_num]) == bool and data_2d == None:
#             self.dataset_trials_filtered[trial_num] = self.load_trial_filter_step(
#                 trial_num).compute()
#         if data_2d is None:
#             data_2d = reshape_to_2d_over_time(
#                 self.dataset_trials_filtered[trial_num])
#         if self.time_trace_params[
#             "time_trace_type"] == "DeltaF Over F" and self.real_trials:
#             time_trace = calculateDeltaFOverF(roi, data_2d,
#                                               denoise=self.time_trace_params[
#                                                   "denoise"] if self.real_trials else False)
#         else:
#             time_trace = calculateMeanTrace(roi, data_2d,
#                                             denoise=self.time_trace_params[
#                                                 "denoise"] if self.real_trials else False)
#         self.time_traces[roi_num - 1][trial_num] = time_trace

def get_time_trace(self, num, trial=None, trace_type="Mean Florescence"):
    """
    Returns the time trace for a certain roi over all currently selected trials
    Parameters
    ----------
    num : int
        ROI num starts at 1 to num rois

    Returns
    -------
    np.ndarray of the time trace
    """
    if not self.real_trials:
        try:
            if len(self.time_traces[trace_type][num - 1]) == 1 and type(
                    self.time_traces[trace_type][num - 1][0]) != bool:
                return self.time_traces[trace_type][num - 1][0]
            else:
                return False

        except IndexError:
            return False

    if (trial == None):
        num = num - 1
        if self.real_trials:
            output = np.ndarray(shape=(0))
            for trial_num in self.trials_loaded_time_trace_indices:
                output = np.hstack(
                    [output, self.time_traces[trace_type][num][trial_num]])
        else:
            return self.time_traces[num]
    else:
        num = num - 1
        output = self.time_traces[num][trial]
    return output
