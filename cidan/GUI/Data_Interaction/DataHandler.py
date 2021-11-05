import logging
from functools import reduce
from math import ceil

import dask
import numpy as np

from cidan.GUI.Data_Interaction.data_handler_functions.change_param import *
from cidan.GUI.Data_Interaction.data_handler_functions.export import \
    export_overlapping_rois, save_image, export, calculate_statistics
from cidan.GUI.Data_Interaction.data_handler_functions.gen_rois_functions import \
    gen_roi_display_variables, genRoiFromPoint, calculate_roi_extraction, \
    gen_class_display_variables
from cidan.GUI.Data_Interaction.data_handler_functions.load_filter_functions import \
    calculate_dataset, transform_data_to_zarr, reset_data, load_dataset, \
    compute_trial_filter_step, load_mask
from cidan.GUI.Data_Interaction.data_handler_functions.save_functions import \
    load_param_json, save_new_param_json
from cidan.GUI.Data_Interaction.data_handler_functions.time_trace_functions import \
    calculate_time_traces, get_time_trace
from cidan.LSSC.SpatialBox import SpatialBox
from cidan.LSSC.functions.data_manipulation import auto_crop as auto_crop_func
from cidan.LSSC.functions.eigen import loadEigenVectors
from cidan.LSSC.functions.pickle_funcs import *
from cidan.TimeTrace.deltaFOverF import calculateDeltaFOverF
from cidan.TimeTrace.mean import calculateMeanTrace, neuropil

logger1 = logging.getLogger("cidan.DataHandler")


def connected_components_graph(A_csr, param, return_labels):
    pass

# NOW rewrite the original initiation function to include new stuff
class DataHandler:
    """
    Interacts with the algorithm and stores the current image data

    This class is initialized whenever data is loaded into the program, it handles
    saving the parameter file, running each of the algorithms in timetrace and LSSC.

    Attributes
    ----------
    global_params : Dict
        This saves the general params of for the dataset, basically whether parameters
        have changed so that the dataset, filters, boxes, eigen vectors, or
        roi extraction need to be recalculated


    """
    # x is for roi data, y is for neuropil data, z is roi data denoised
    time_trace_possibilities_functions = {
        "Mean Florescence Denoised": lambda x, y: calculateMeanTrace(x, y),
        "Mean Florescence": lambda x, y: calculateMeanTrace(x, y),
        "DeltaF Over F Denoised": lambda x, y: calculateDeltaFOverF(x, y),
        "DeltaF Over F": lambda x, y: calculateDeltaFOverF(x, y),
        "Mean Floresence Denoised (Neuropil Corrected)": lambda x,
                                                                y: calculateMeanTrace(x,
                                                                                      y,
                                                                                      sub_neuropil=True),
        "Mean Floresence  (Neuropil Corrected)": lambda x, y: calculateMeanTrace(x,
                                                                                 y,

                                                                                 sub_neuropil=True),
        "DeltaF Over F  (Neuropil Corrected)": lambda x, y: calculateDeltaFOverF(x,
                                                                                 y,

                                                                                 sub_neuropil=True),
        "DeltaF Over F Denoised (Neuropil Corrected)": lambda x,
                                                              y: calculateDeltaFOverF(x,
                                                                                      y,

                                                                                      sub_neuropil=True),
        "Neuropil": lambda x, y: neuropil(x, y)
    }
    _color_list = [(218, 67, 34),
                   (132, 249, 22), (22, 249, 140), (22, 245, 249),
                   (22, 132, 249), (224, 22, 249), (249, 22, 160)]
    _base_class_construct = {"Unassigned": {"color": (150, 150, 150), "rois": [],
                                            "name": "Unassigned", "editable": False},
                             "Base": {"color": _color_list[0], "rois": [],
                                      "name": "Base", "editable": True},
                             "Important": {"color": _color_list[1], "rois": [],
                                           "name": "important", "editable": True}}
    _global_params_default = {
        "save_intermediate_steps": True,
        "need_recalc_dataset_params": True,
        "need_recalc_filter_params": True,
        "need_recalc_box_params": True,
        "need_recalc_eigen_params": True,
        "need_recalc_roi_extraction_params": True,
        "load_into_mem": True,
    }

    _dataset_params_default = {
        "dataset_folder_path": "",
        "trials_loaded": [],
        "trials_all": [],
        "mask_path": "",
        "single_file_mode": False,
        "original_folder_trial_split": "",
        "slice_stack": False,
        "slice_every": 3,
        "slice_start": 0,
        "crop_stack": False,
        "crop_x": [0, 0],
        "crop_y": [0, 0],
        "trial_split": False,
        "trial_length": 400,
        "auto_crop": False
    }

    _filter_params_default = {
        "median_filter": False,
        "median_filter_size": 3,
        "z_score": False,
        "hist_eq": False,
        "localSpatialDenoising": True,
        "pca": False,
        "pca_threshold": .97

    }
    _filter_params_default_widefield = {
        "median_filter": False,
        "median_filter_size": 3,
        "z_score": False,
        "hist_eq": False,
        "localSpatialDenoising": True,
        "pca": False,
        "pca_threshold": .97

    }
    _box_params_default = {
        "total_num_time_steps": 1,
        "total_num_spatial_boxes": 1,
        "spatial_overlap": 40
    }
    _eigen_params_default = {
        "eigen_vectors_already_generated": False,
        "num_eig": 50,
        "normalize_w_k": 32,
        "metric": "l2",
        "knn": 50,
        "accuracy": 75,
        "eigen_accuracy": 8,
        "connections": 40

    }
    _roi_extraction_params_default = {
        "elbow_threshold_method": True,
        "elbow_threshold_value": .95,
        "eigen_threshold_method": True,
        "eigen_threshold_value": .9,
        "num_eigen_vector_select": 1,
        "merge_temporal_coef": .9,
        "roi_size_min": 30,
        "roi_size_max": 600,
        "merge": True,
        "num_rois": 60,
        "fill_holes": True,
        "refinement": True,
        "max_iter": 100,
        "roi_circ_threshold": 0,
        "roi_eccentricity_limit": .9,
        "local_max_method": False,
        "area_stop_threshold": .95

    }
    _roi_extraction_params_default_widefield = {
        "elbow_threshold_method": True,
        "elbow_threshold_value": .95,
        "eigen_threshold_method": True,
        "eigen_threshold_value": .9,
        "num_eigen_vector_select": 1,
        "merge_temporal_coef": .9,
        "roi_size_min": 30,
        "roi_size_max": 600,
        "merge": True,
        "num_rois": 60,
        "fill_holes": True,
        "refinement": True,
        "max_iter": 100,
        "roi_circ_threshold": 0,
        "roi_eccentricity_limit": .9,
        "local_max_method": False,
        "area_stop_threshold": .95

    }
    _time_trace_params_default = {
        "min_neuropil_pixels": 25
    }

    def __init__(self, data_path, save_dir_path, save_dir_already_created, trials=[],
                 parameter_file=False, load_into_mem=True, auto_crop=False,
                 widefield=False, mask_path=None):
        """
        Initializes the object
        Parameters
        ----------
        data_path : str
            This is the path towards the folder containing the data files/folder null
            if save dir is already created
        save_dir_path : str
            save folder location
        save_dir_already_created : bool
            whether to load all the info from the save dir, if this is true data_path
            and trials don't matter
        trials : List[str]
            The list of either names of folders that contain many tiff files or a
            list of tiff files
        """
        self.parameter_file_name = "parameters.json"
        # TODO add loaded trials and all trials parameter here
        # TODO make sure if trial list includes files that aren't valid it works
        self.color_list = DataHandler._color_list

        self.save_dir_path = save_dir_path
        self.time_trace_possibilities_functions = DataHandler.time_trace_possibilities_functions
        if widefield:
            self.time_trace_possibilities_functions = {
                key: DataHandler.time_trace_possibilities_functions[key] for key in
                self.time_trace_possibilities_functions.keys() if "Neuropil" not in key}
        self.rois_loaded = False  # whether roi variables have been created
        self.rois_update_needed = True
        self.widefield = widefield
        self.image_data_mask_flat = None
        if parameter_file:
            self.parameter_file_name = os.path.basename(parameter_file)
            self.save_dir_path = os.path.dirname(parameter_file)
            self.classes = DataHandler._base_class_construct
            self.create_new_save_dir()
            valid = self.load_param_json()
            self.global_params = DataHandler._global_params_default.copy()
            self.global_params["load_into_mem"] = load_into_mem
            if self.dataset_params["single_file_mode"] and not load_into_mem:
                if not os.path.isdir(
                        os.path.join(self.save_dir_path, "temp_files/dataset.zarr")):
                    self.transform_data_to_zarr()
            # these indices are used for which trials to use for roi extract
            self._trials_loaded_indices = [num for num, x in enumerate(self.trials_all)
                                           if x in self.trials_loaded]
            # these are used to determine which trials to calculate time_traces for
            self.trials_loaded_time_trace_indices = [num for num, x in
                                                     enumerate(self.trials_all)
                                                     if x in self.trials_loaded]

            if self.widefield:
                self.load_mask(self.mask_path)
            self.load_dataset(
                [os.path.join(self.dataset_params["dataset_folder_path"], x) for
                 x
                 in self.trials_loaded])
            self.reset_data()
            dask.compute(*self.dataset_trials_filtered_loaded)
            self.global_params["need_recalc_filter_params"] = False
            self.global_params[
                "need_recalc_dataset_params"] = False




        elif save_dir_already_created:  # this loads everything from the save dir
            valid = self.load_param_json()
            self.classes = DataHandler._base_class_construct
            self.time_trace_params = DataHandler._time_trace_params_default.copy()
            if self.dataset_params["single_file_mode"] and not self.load_into_mem:
                if not os.path.isdir(
                        os.path.join(self.save_dir_path, "temp_files/dataset.zarr")):
                    self.transform_data_to_zarr()
            # time trace params are not currently saved

            # these indices are used for which trials to use for roi extract
            self._trials_loaded_indices = [num for num, x in enumerate(self.trials_all)
                                           if x in self.trials_loaded]
            # these are used to determine which trials to calculate time_traces for
            self.trials_loaded_time_trace_indices = [num for num, x in
                                                     enumerate(self.trials_all)
                                                     if x in self.trials_loaded]
            if self.widefield:
                self.load_mask(self.mask_path)
            # this loads the dataset and calculate the specified filters
            if not self.load_into_mem:
                self.load_dataset(
                    [os.path.join(self.dataset_params["dataset_folder_path"], x) for
                     x
                     in self.trials_loaded])
                self.reset_data()

                self.dataset_trials_filtered[
                    self._trials_loaded_indices[0]].compute()

                print("Loaded Previous Session")

            else:
                temp = self.global_params["need_recalc_box_params"]

                self.load_dataset(
                    [os.path.join(self.dataset_params["dataset_folder_path"], x) for x
                     in self.trials_loaded])
                self.reset_data()

                self.dataset_trials_filtered[
                    self._trials_loaded_indices[0]].compute()

                self.global_params["need_recalc_box_params"] = temp
            # if there are ROIs saved in the save dir load them and calculate time
            # traces


            if self.rois_exist:
                try:
                    self.load_rois()

                    self.rois_loaded = True
                except:
                    print("ROI loading Failed")
                self.calculate_time_traces()
            if not valid:
                raise FileNotFoundError("Save directory not valid")
        else:
            # start with the default values specified in this class

            self.global_params = DataHandler._global_params_default.copy()
            self.global_params["load_into_mem"] = load_into_mem
            self.dataset_params = DataHandler._dataset_params_default.copy()
            self.filter_params = DataHandler._filter_params_default.copy()
            self.box_params = DataHandler._box_params_default.copy()
            self.box_params_processed = DataHandler._box_params_default.copy()
            self.roi_extraction_params = DataHandler._roi_extraction_params_default.copy()
            if widefield:
                self.roi_extraction_params = DataHandler._roi_extraction_params_default_widefield.copy()

            self.time_trace_params = DataHandler._time_trace_params_default.copy()
            self.classes = DataHandler._base_class_construct
            self.dataset_params["dataset_folder_path"] = data_path
            self.dataset_params["trials_loaded"] = trials
            self.trials_loaded = trials
            self.dataset_params["mask_path"] = mask_path
            if not os.path.isdir(
                    os.path.join(data_path, trials[0])) and len(
                self.trials_loaded) == 1:
                self.dataset_params[
                    "single_file_mode"] = True  # this is different from single dataset mode which is if the len(self.data_list)==1
            if len(self.trials_loaded) > 1:
                self.dataset_params["original_folder_trial_split"] = trials

            if len(self.trials_loaded) == 1:
                self.dataset_params["trial_split"] = True
                self.dataset_params["original_folder_trial_split"] = [trials[0]]
            # these are all files in the data_path directory if the user wants to
            # calculate time traces than for more trials than they originally loaded
            self.trials_all = sorted(os.listdir(data_path))
            if not os.path.isdir(os.path.join(data_path, trials[0])):
                self.trials_all = [x for x in self.trials_all if ".tif" in x]
            self.dataset_params["trials_all"] = self.trials_all

            self.box_params["total_num_time_steps"] = len(trials)
            self.eigen_params = DataHandler._eigen_params_default.copy()

            valid = self.create_new_save_dir()
            if not valid:
                raise FileNotFoundError("Please chose an empty directory for your " +
                                        "save directory")
            if self.dataset_params["single_file_mode"] and not load_into_mem:
                self.transform_data_to_zarr()
            if self.widefield:
                self.load_mask(self.mask_path)

            self.load_dataset(
                [os.path.join(self.dataset_params["dataset_folder_path"], x) for x in
                 self.trials_loaded])

            self._trials_loaded_indices = [num for num, x in enumerate(self.trials_all)
                                           if x in self.trials_loaded]
            self.trials_loaded_time_trace_indices = [num for num, x in
                                                     enumerate(self.trials_all)
                                                     if x in self.trials_loaded]

            if auto_crop:
                self.dataset_params["auto_crop"] = False
                self.dataset_params["crop_x"], self.dataset_params[
                    "crop_y"] = auto_crop_func(self.dataset_list)
                self.dataset_params["crop_stack"] = True
                self.reset_data()

                self.dataset_trials_filtered[self._trials_loaded_indices[0]].compute()
            else:
                self.reset_data()

                self.dataset_trials_filtered[self._trials_loaded_indices[0]].compute()
                self.dataset_params["crop_x"] = [0, self.shape[0]]
                self.dataset_params["crop_y"] = [0, self.shape[1]]

            self.save_new_param_json()

    def __del__(self):
        try:
            for x in list(self.__dict__.items()):
                self.__dict__[x[0]] = None
        except TypeError:
            pass

    @property
    def mask_path(self):
        return self.dataset_params["mask_path"]

    @property
    def dataset_trials_filtered_loaded(self):
        """
        Returns the filtered data that is currently loaded
        """
        return [self.dataset_trials_filtered[x] for x in self._trials_loaded_indices]

    @property
    def dataset_trials_loaded(self):
        """
        Returns the unfiltered data that is currently loaded
        """
        return [self.dataset_trials[x] for x in self._trials_loaded_indices]

    @property
    def param_path(self):
        """
        Returns the path to the parameter file
        """
        return os.path.join(self.save_dir_path, self.parameter_file_name)

    @property
    def eigen_vectors_exist(self):
        """
        Returns if the eigen vectors are saved for the current settings
        """

        eigen_dir = os.path.join(self.save_dir_path, "eigen_vectors")

        file_names = [
            "eigen_vectors_box_{}_{}.pickle".format(spatial_box_num, time_box_num)
            for spatial_box_num in range(self.box_params["total_num_spatial_boxes"])
            for time_box_num in range(self.box_params["total_num_time_steps"])]
        return all(pickle_exist(x, output_directory=eigen_dir) for x in file_names)

    @property
    def auto_crop(self):
        return self.dataset_params["auto_crop"]

    @property
    def load_into_mem(self):
        return self.global_params["load_into_mem"]
    @property
    def rois_exist(self):
        """
        Return if the roi save file exists
        """
        return not (self.global_params["need_recalc_eigen_params"] or
                    self.global_params[
                        "need_recalc_roi_extraction_params"] or self.global_params[
                        "need_recalc_box_params"]) and pickle_exist("rois",
                                                                    output_directory=self.save_dir_path)

    @property
    def load_into_mem(self):
        return self.global_params["load_into_mem"]
    @property
    def single_dataset_mode(self):
        # basically whether there are original trials
        return len(self.dataset_params[
                       "original_folder_trial_split"]) == 1
    def load_rois(self):
        """
        Loads the ROIs if they exist and then generates the other variables associated
        with them
        """
        if pickle_exist("rois", output_directory=self.save_dir_path):
            rois = pickle_load("rois", output_directory=self.save_dir_path)
            if type(rois) == dict:
                self.rois_dict = rois
            else:
                self.rois_dict = {(num + 1): {"pixels": roi, "index": num} for num, roi
                                  in enumerate(rois)}
            try:
                self.classes = pickle_load("classes", output_directory=self.classes)
            except:
                self.classes = {"Unassigned": {"color": (150, 150, 150), "rois": [],
                                               "name": "Unassigned", "editable": False},
                                "Base": {"color": self.color_list[0], "rois": [],
                                         "name": "Base", "editable": True},
                                "Important": {"color": self.color_list[1], "rois": [],
                                              "name": "important", "editable": True}}

            self.gen_roi_display_variables()

    def save_rois(self, rois):
        """
        Saves the rois to files
        Parameters
        ----------
        rois : List[np.ndarray]
            A list pixels of each roi, each pixel is its 1d cordinate

        """
        if os.path.isdir(self.save_dir_path):
            pickle_save(self.rois_dict, "rois", output_directory=self.save_dir_path)
            pickle_save(self.classes, "classes", output_directory=self.save_dir_path)
    def load_param_json(self):
        """
        Loads the parameter json file and saves it to all the parameter values
        """
        return load_param_json(self)
    def save_new_param_json(self):
        """
        Saves the parameters to the parameter file
        """
        return save_new_param_json(self)

    def create_new_save_dir(self):
        """
        Creates a new save directory
        """
        try:
            if not os.path.isdir(self.save_dir_path):
                os.mkdir(self.save_dir_path)
            eigen_vectors_folder_path = os.path.join(self.save_dir_path,
                                                     "eigen_vectors/")

            if not os.path.isdir(eigen_vectors_folder_path):
                os.mkdir(eigen_vectors_folder_path)
            embedding_images_path = os.path.join(self.save_dir_path,
                                                 "embedding_norm_images/")
            if not os.path.isdir(embedding_images_path):
                os.mkdir(embedding_images_path)
            eigen_vectors_folder_path = os.path.join(self.save_dir_path,
                                                     "temp_files/")

            if not os.path.isdir(eigen_vectors_folder_path):
                os.mkdir(eigen_vectors_folder_path)
            return True
        except:
            raise FileNotFoundError("Couldn't create folder please try again")

    def change_global_param(self, param_name, new_value):
        return change_global_param(self, param_name, new_value)

    def change_dataset_param(self, param_name, new_value):
        return change_dataset_param(self, param_name, new_value)

    def change_filter_param(self, param_name, new_value):
        return change_filter_param(self, param_name, new_value)

    def change_box_param(self, param_name, new_value):
        return change_box_param(self, param_name, new_value)

    def change_eigen_param(self, param_name, new_value):
        return change_eigen_param(self, param_name, new_value)

    def change_roi_extraction_param(self, param_name, new_value):
        return change_roi_extraction_param(self, param_name, new_value)

    def transform_data_to_zarr(self):
        return transform_data_to_zarr(self)

    def update_trial_list(self):
        if not self.dataset_params["trial_split"] and self.single_dataset_mode:
            self.trials_loaded = ["0"]
            self.trials_all = self.trials_loaded.copy()
            self.box_params["total_num_time_steps"] = len(self.trials_loaded)
        elif self.single_dataset_mode:
            self.trials_loaded = [str(x) for x in
                                  range(ceil(self.dataset_list[0].shape[0] /
                                             self.dataset_params[
                                                 "trial_length"]))]
            self.trials_all = self.trials_loaded.copy()
            self.box_params["total_num_time_steps"] = len(self.trials_loaded)
        elif not self.single_dataset_mode:
            self.trials_loaded = self.dataset_params["original_folder_trial_split"]
            self.trials_all = self.dataset_params["original_folder_trial_split"]
            self.box_params["total_num_time_steps"] = len(self.trials_loaded)
        if self.single_dataset_mode:
            self._trials_loaded_indices = [num for num, x in
                                           enumerate(self.trials_all)
                                           if x in self.trials_loaded]
            self.trials_loaded_time_trace_indices = [num for num, x in
                                                     enumerate(self.trials_all)
                                                     if x in self.trials_loaded]
    def calculate_dataset(self, ) -> np.ndarray:
        """
        Loads each trial, applying crop and slicing, sets them to self.dataset_trials

        """
        return calculate_dataset(self)
    def compute_trial_filter_step(self, trial_num,loaded_num,dataset=False,save_data=True ):
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
        return compute_trial_filter_step(self, trial_num, loaded_num, dataset,
                                         save_data=save_data)

    # def load_trial_dataset_step(self, trial_num):
    #     """
    #     Delayed function step that loads a single trial
    #     Parameters
    #     ----------
    #     trial_num : int
    #         trial num to load in trials_all
    #
    #     Returns
    #     -------
    #     Trial as a np.ndarray
    #     """
    #     return load_trial_dataset_step(self, trial_num)
    #
    # def load_trial_filter_step(self, trial_num, dataset=False, loaded_num=False):
    #     """
    #     Delayed function step that applies filter to a single trial
    #     Parameters
    #     ----------
    #     trial_num : int
    #         trial num to load in trials_all
    #     dataset : ndarray
    #         Dataset to process if false load dataset
    #
    #     Returns
    #     -------
    #     Filtered trial as a np.ndarray
    #     """
    #     return load_trial_filter_step(self, trial_num, dataset, loaded_num)
    #
    # def calculate_filters(self, trial=0,progress_signal=None, auto_crop=False):
    #     """
    #     Applies filter to each trial, sets them to self.dataset_trials_filtered
    #
    #     Returns
    #     -------
    #     A list of filtered trials
    #     """
    #     return calculate_filters(self, trial,progress_signal, auto_crop)
    def reset_data(self):
        # resets all the data attributes but doesn't reload data itself
        return reset_data(self)

    def load_dataset(self, path):
        # loads a dataset or loads a zarr in
        return load_dataset(self, path)

    def load_mask(self, path):
        return load_mask(self, path)

    @property
    def real_trials(self):
        return not (self.dataset_params["single_file_mode"] or self.dataset_params[
            "trial_split"] or len(self.trials_loaded) == 1)

    def calculate_roi_extraction(self, progress_signal=None):
        """
        Extracts Rois and sets them to self.rois
        """
        return calculate_roi_extraction(self, progress_signal)

    def gen_roi_display_variables(self):
        """
        Generates the other variables associated with displaying ROIs including class ones
        """
        return gen_roi_display_variables(self)

    def gen_class_display_variables(self):
        """
        Generates the class variables associated with displaying ROIs
        """
        return gen_class_display_variables(self)

    def calculate_time_traces(self, report_progress=None):
        """
        Calculates the time traces for every roi in self.rois
        """
        return calculate_time_traces(self, report_progress)

    @property
    def rois(self):

        if self.rois_update_needed:
            keys = list(self.rois_dict.keys())
            for num, x in enumerate(keys):
                self.rois_dict[x]["index"] = num
            self._rois = [self.rois_dict[x]["pixels"] for x in keys]
            self.roi_index_backward = {num: x for num, x in enumerate(keys)}
            self.rois_update_needed = False
        return self._rois

    def assign_roi_class(self, key, new_class, input_key=True):
        if not input_key:
            key = self.roi_index_backward[key]
        self.remove_roi_from_all_classes(key, input_key=True)
        self.classes[new_class]["rois"].append(key)
        self.gen_class_display_variables()

    def remove_roi_from_all_classes(self, key, input_key=True):
        if not input_key:
            key = self.roi_index_backward[key]
        for class_name in self.classes:
            if key in self.classes[class_name]["rois"]:
                self.classes[class_name]["rois"].remove(key)

    def delete_roi(self, key, input_key=True):
        if not input_key:
            key = self.roi_index_backward[key]

        for x in self.time_traces.keys():
            self.time_traces[x].pop(self.rois_dict[key]["index"])
        self.roi_time_trace_need_update.pop(self.rois_dict[key]["index"])
        del self.rois_dict[key]

        self.rois_update_needed = True
        self.gen_roi_display_variables()

    def add_new_roi(self, roi_pixels, update=True):
        key = max(list(self.rois_dict.keys())) + 1
        self.rois_dict[key] = {"pixels": roi_pixels}
        self.rois_update_needed = True
        if update:
            self.gen_roi_display_variables()
        self.roi_time_trace_need_update.append(True)
        return key

    def update_roi(self, key, new_pixels, input_key=True):
        if not input_key:
            key = self.roi_index_backward[key]
        self.rois_dict[key]["pixels"] = new_pixels
        self.rois_update_needed = True
        self.gen_roi_display_variables()

    def merge_rois(self, keys, input_key=True):
        if not input_key:
            keys = [self.roi_index_backward[key] for key in keys]
        pixels = set(list(reduce(lambda x, y: x + y,
                                 [list(self.rois_dict[x]["pixels"]) for x in keys])))
        for key in keys:
            self.rois_dict.pop(key, "test")
        index = max(list(self.rois_dict.keys())) + 1
        self.rois_dict[index] = {"pixels": np.array(list(pixels))}
        self.rois_update_needed = True
        self.gen_roi_display_variables()
        return index

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

        return get_time_trace(self, num, trial, trace_type)

    def calculate_filters(self, progress_signal=None):
        if self.global_params["need_recalc_filter_params"] or self.global_params[
            "need_recalc_dataset_params"] or \
            not hasattr(self, "dataset_trials_filtered"):
            self.reset_data()
            self.save_new_param_json()
            self.dataset_trials_filtered[self._trials_loaded_indices[0]].compute()
            self.global_params["need_recalc_filter_params"] =False
            self.global_params[
                "need_recalc_dataset_params"] =False
    def update_selected_trials(self, selected_trials):
        """
        Updates the loaded trials so don't have unnecessary loaded trials
        Parameters
        ----------
        selected_trials
            trials to be currently loaded
        """
        new_selected = [x for x, y in enumerate(self.trials_all) if
                        y in selected_trials]
        for trial_num, _ in enumerate(self.trials_all):
            if trial_num in new_selected and trial_num not in self.trials_loaded_time_trace_indices:
                self.dataset_trials_filtered[trial_num] = self.load_trial_filter_step(
                    trial_num).compute()
                self.trials_loaded_time_trace_indices.append(trial_num)
            if trial_num not in new_selected and trial_num in self.trials_loaded_time_trace_indices \
                    and trial_num not in self._trials_loaded_indices:
                self.dataset_trials_filtered[trial_num] = False

            if trial_num not in new_selected and trial_num in self.trials_loaded_time_trace_indices:
                self.trials_loaded_time_trace_indices.remove(trial_num)

    def get_eigen_vector(self, box_num, vector_num, trial_num):
        vector = loadEigenVectors(spatial_box_num=box_num, time_box_num=trial_num,
                                  save_dir=self.save_dir_path).compute()
        spatial_box = SpatialBox(box_num,
                                 total_boxes=self.box_params["total_num_spatial_boxes"],
                                 image_shape=self.shape,
                                 spatial_overlap=self.box_params[
                                                     "spatial_overlap"] // 2)
        return vector[:, vector_num].reshape(spatial_box.shape)

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
        return genRoiFromPoint(self, point, growth_factor)

    def calculate_statistics(self):
        return calculate_statistics(self)

    def export(self, matlab=False, background_images=["max", "mean", "eigen_norm", "blank"],
               color_maps=['gray']):
        return export(self, matlab, background_images, color_maps)

    def save_image(self, image, path):
        return save_image(self, image, path)

    def delete_roi_vars(self):
        # self.rois = []
        self.rois_dict = {}
        self.rois_update_needed = True
        for key in self.classes.keys():
            self.classes[key]["rois"] = []
        self.rois_loaded = False
        self.roi_max_cord_list = None
        self.roi_min_cord_list = None
        self.pixel_with_rois_flat = None
        self.pixel_with_rois_color_flat = None
        self.time_traces = []
        self.edge_roi_image_flat = None
        self.pixel_with_rois_color = None
        self.eigen_norm_image = None

    def export_overlapping_rois(self, current_roi_num_1, current_roi_num_2):
        """THis is only used in dev version it allows the user to export time trace of all pixels in this roi and in overlapping rois"""
        export_overlapping_rois(self, current_roi_num_1, current_roi_num_2)
