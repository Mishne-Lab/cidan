import json
import json
import logging
from functools import reduce
from math import ceil

import dask
import numpy as np
import zarr
from PIL import Image
from dask import delayed
from scipy.io import savemat
from skimage import feature
from tifffile import tifffile

from cidan.LSSC.SpatialBox import SpatialBox
from cidan.LSSC.functions.data_manipulation import load_filter_tif_stack, filter_stack, \
    reshape_to_2d_over_time, pixel_num_to_2d_cord, applyPCA
from cidan.LSSC.functions.eigen import loadEigenVectors
from cidan.LSSC.functions.pickle_funcs import *
from cidan.LSSC.functions.progress_bar import printProgressBarFilter, printProgressBar
from cidan.LSSC.functions.roi_extraction import roi_extract_image, combine_rois
from cidan.LSSC.functions.roi_filter import filterRoiList
from cidan.LSSC.functions.spatial_footprint import classify_components_ep
from cidan.LSSC.process_data import process_data
from cidan.TimeTrace.deltaFOverF import calculateDeltaFOverF
from cidan.TimeTrace.mean import calculateMeanTrace, neuropil
from cidan.TimeTrace.neuropil import calculate_nueropil
from cidan.TimeTrace.waveletDenoise import waveletDenoise

logger1 = logging.getLogger("cidan.DataHandler")


def connected_components_graph(A_csr, param, return_labels):
    pass


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
        "Mean Florescence Denoised": lambda x, y, z: calculateMeanTrace(x, y, z,
                                                                        denoise=True),
        "Mean Florescence": lambda x, y, z: calculateMeanTrace(x, y, z, denoise=False),
        "DeltaF Over F Denoised": lambda x, y, z: calculateDeltaFOverF(x, y, z,
                                                                       denoise=True),
        "DeltaF Over F": lambda x, y, z: calculateDeltaFOverF(x, y, z, denoise=False),
        "Mean Floresence Denoised (Neuropil Corrected)": lambda x,
                                                                y,
                                                                z: calculateMeanTrace(x,
                                                                                      y,
                                                                                      z,
                                                                                      denoise=True,
                                                                                      sub_neuropil=True),
        "Mean Floresence  (Neuropil Corrected)": lambda x, y, z: calculateMeanTrace(x,
                                                                                    y,
                                                                                    z,
                                                                                    denoise=False,
                                                                                    sub_neuropil=True),
        "DeltaF Over F  (Neuropil Corrected)": lambda x, y, z: calculateDeltaFOverF(x,
                                                                                    y,
                                                                                    z,
                                                                                    denoise=False,
                                                                                    sub_neuropil=True),
        "DeltaF Over F Denoised (Neuropil Corrected)": lambda x,
                                                              y,
                                                              z: calculateDeltaFOverF(x,
                                                                                      y,
                                                                                      z,
                                                                                      denoise=True,
                                                                                      sub_neuropil=True),
        "Neuropil": lambda x, y, z: neuropil(x, y, z)
    }
    _global_params_default = {
        "save_intermediate_steps": True,
        "need_recalc_dataset_params": True,
        "need_recalc_filter_params": True,
        "need_recalc_box_params": True,
        "need_recalc_eigen_params": True,
        "need_recalc_roi_extraction_params": True,
        "load_into_mem": False,
        "num_threads": 8
    }

    _dataset_params_default = {
        "dataset_folder_path": "",
        "trials_loaded": [],
        "trials_all": [],
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
    _box_params_default = {
        "total_num_time_steps": 1,
        "total_num_spatial_boxes": 1,
        "spatial_overlap": 40
    }
    _eigen_params_default = {
        "eigen_vectors_already_generated": False,
        "num_eig": 50,
        "normalize_w_k": 25,
        "metric": "l2",
        "knn": 32,
        "accuracy": 75,
        "eigen_accuracy": 8,
        "connections": 40

    }
    _roi_extraction_params_default = {
        "elbow_threshold_method": True,
        "elbow_threshold_value": .95,
        "eigen_threshold_method": True,
        "eigen_threshold_value": .1,
        "num_eigen_vector_select": 5,
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
        "local_max_method": False

    }
    _time_trace_params_default = {
        "min_neuropil_pixels": 25
    }

    def __init__(self, data_path, save_dir_path, save_dir_already_created, trials=[],
                 parameter_file=False, load_into_mem=True):
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
        self.color_list = [(218, 67, 34),
                           (132, 249, 22), (22, 249, 140), (22, 245, 249),
                           (22, 132, 249), (224, 22, 249), (249, 22, 160)]

        self.save_dir_path = save_dir_path
        self.time_trace_possibilities_functions = DataHandler.time_trace_possibilities_functions
        self.rois_loaded = False  # whether roi variables have been created
        if parameter_file:
            self.parameter_file_name = os.path.basename(parameter_file)
            self.save_dir_path = os.path.dirname(parameter_file)
            self.create_new_save_dir()
            valid = self.load_param_json()
            self.global_params = DataHandler._global_params_default.copy()
            if self.dataset_params["single_file_mode"]:
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


        elif save_dir_already_created:  # this loads everything from the save dir
            valid = self.load_param_json()
            self.time_trace_params = DataHandler._time_trace_params_default.copy()
            if self.dataset_params["single_file_mode"]:
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
            # this loads the dataset and calculate the specified filters
            if not self.load_into_mem:
                try:
                    self.load_data()
                    print("Loaded Previous Session")
                except:
                    self.__delattr__("dataset_trials_filtered")
                    temp = self.global_params["need_recalc_box_params"]
                    self.calculate_filters()
                    self.global_params["need_recalc_box_params"] = temp
            else:
                temp = self.global_params["need_recalc_box_params"]
                self.calculate_filters()
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
            self.time_trace_params = DataHandler._time_trace_params_default.copy()

            self.dataset_params["dataset_folder_path"] = data_path
            self.dataset_params["trials_loaded"] = trials
            self.trials_loaded = trials
            if not os.path.isdir(
                    os.path.join(data_path, trials[0])) and len(
                self.trials_loaded) == 1:
                self.dataset_params["single_file_mode"] = True
            if len(self.trials_loaded) == 1:
                self.dataset_params["trial_split"] = True
                self.dataset_params["original_folder_trial_split"] = trials[0]
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
            if self.dataset_params["single_file_mode"]:
                self.transform_data_to_zarr()
            self.time_traces = []
            self._trials_loaded_indices = [num for num, x in enumerate(self.trials_all)
                                           if x in self.trials_loaded]
            self.trials_loaded_time_trace_indices = [num for num, x in
                                                     enumerate(self.trials_all)
                                                     if x in self.trials_loaded]
            self.save_new_param_json()

    def __del__(self):
        try:
            for x in list(self.__dict__.items()):
                self.__dict__[x[0]] = None
        except TypeError:
            pass

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
    def rois_exist(self):
        """ƒc
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

    def load_rois(self):
        """
        Loads the ROIs if they exist and then generates the other variables associated
        with them
        """
        if pickle_exist("rois", output_directory=self.save_dir_path):
            self.rois = pickle_load("rois", output_directory=self.save_dir_path)
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
            pickle_save(rois, "rois", output_directory=self.save_dir_path)

    def load_param_json(self):
        """
        Loads the parameter json file and saves it to all the parameter values
        """

        try:
            with open(self.param_path, "r") as f:
                all_params = json.loads(f.read())
            self.global_params = all_params["global_params"]
            self.dataset_params = all_params["dataset_params"]
            self.filter_params = all_params["filter_params"]
            self.box_params = all_params["box_params"]
            self.box_params_processed = all_params["box_params"].copy()
            self.eigen_params = all_params["eigen_params"]
            self.time_trace_params = all_params["time_trace_params"]
            self.roi_extraction_params = all_params["roi_extraction_params"]
            self.trials_loaded = self.dataset_params["trials_loaded"]
            self.trials_all = self.dataset_params["trials_all"]
            return True
        except KeyError:
            raise KeyError("Please Choose a valid parameter file")
        except FileNotFoundError:
            raise FileNotFoundError("Can't find parameter file")
        except NameError:
            raise FileNotFoundError("Can't find parameter file")

    def save_new_param_json(self):
        """
        Saves the parameters to the parameter file
        """
        try:
            with open(self.param_path, "w") as f:
                all_params = {
                    "global_params": self.global_params,
                    "dataset_params": self.dataset_params,
                    "filter_params": self.filter_params,
                    "box_params": self.box_params_processed,
                    "eigen_params": self.eigen_params,
                    "roi_extraction_params": self.roi_extraction_params,
                    "time_trace_params": self.time_trace_params
                }
                f.truncate(0)
                f.write(json.dumps(all_params))
        except:
            raise FileNotFoundError("Error saving parameters, please restart software")
        pass

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
        """
        Used to change a param in global params area.
        Parameters
        ----------
        param_name : str
            name of parameter to change
        new_value : ?
            the new value to the parameter

        Returns
        -------
        True if successful
        """
        if param_name in self.global_params:
            self.global_params[param_name] = new_value
            self.save_new_param_json()
            return True
        else:
            return False

    def change_dataset_param(self, param_name, new_value):
        """
        Used to change a param in dataset params area.
        Parameters
        ----------
        param_name : str
            name of parameter to change
        new_value : ?
            the new value to the parameter

        Returns
        -------
        True if successful
        """
        if param_name in self.dataset_params:
            self.dataset_params[param_name] = new_value
            self.global_params["need_recalc_dataset_params"] = True
            self.global_params["need_recalc_box_params"] = True
            self.global_params["need_recalc_eigen_params"] = True
            # self.save_new_param_json()
            return True
        else:
            return False

    def change_filter_param(self, param_name, new_value):
        """
        Used to change a param in filter params area.
        Parameters
        ----------
        param_name : str
            name of parameter to change
        new_value : ?
            the new value to the parameter

        Returns
        -------
        True if successful
        """
        if param_name in self.filter_params:
            self.filter_params[param_name] = new_value
            self.global_params["need_recalc_filter_params"] = True
            self.global_params["need_recalc_box_params"] = True

            self.global_params["need_recalc_eigen_params"] = True
            # self.save_new_param_json()
            return True
        else:
            return False

    def change_box_param(self, param_name, new_value):
        """
        Used to change a param in box params area.
        Parameters
        ----------
        param_name : str
            name of parameter to change
        new_value : ?
            the new value to the parameter

        Returns
        -------
        True if successful
        """
        if param_name in self.box_params:
            # if param_name == "total_num_spatial_boxes":
            #     assert (int(new_value**.5))**2 == new_value, "Please make sure Number of Spatial Boxes is a square number"
            self.box_params[param_name] = new_value
            self.global_params["need_recalc_box_params"] = True
            self.global_params["need_recalc_eigen_params"] = True
            self.global_params["need_recalc_roi_extraction_params"] = True
            # self.save_new_param_json()
            return True
        else:
            return False

    def change_eigen_param(self, param_name, new_value):
        """
        Used to change a param in eigen params area.
        Parameters
        ----------
        param_name : str
            name of parameter to change
        new_value : ?
            the new value to the parameter

        Returns
        -------
        True if successful
        """
        if param_name in self.eigen_params:
            self.eigen_params[param_name] = new_value
            self.global_params["need_recalc_eigen_params"] = True
            # self.save_new_param_json()
            return True
        else:
            return False

    def change_roi_extraction_param(self, param_name, new_value):
        """
        Used to change a param in roi_extraction params area.
        Parameters
        ----------
        param_name : str
            name of parameter to change
        new_value : ?
            the new value to the parameter

        Returns
        -------
        True if successful
        """
        if param_name in self.roi_extraction_params:
            self.roi_extraction_params[param_name] = new_value
            self.global_params["need_recalc_roi_extraction_params"] = True
            # self.save_new_param_json()
            return True
        else:
            return False

    def transform_data_to_zarr(self):
        image = tifffile.imread(os.path.join(self.dataset_params["dataset_folder_path"],
                                             self.dataset_params[
                                                 "original_folder_trial_split"]))
        z1 = zarr.open(os.path.join(self.save_dir_path,
                                    'temp_files/dataset.zarr'), mode='w',
                       shape=image.shape,
                       chunks=(32, 128, 128), dtype=image.dtype)
        z1[:] = image

    def calculate_dataset(self, ) -> np.ndarray:
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
    def load_trial_dataset_step(self, trial_num):
        """
        Delayed function step that loads a single trial
        Parameters
        ----------
        trial_num : int
            trial num to load in trials_all

        Returns
        -------
        Trial as a np.ndarray
        """
        total_size, stack = load_filter_tif_stack(
            path=os.path.join(self.dataset_params["dataset_folder_path"],
                              self.trials_all[trial_num] if not self.dataset_params[
                                  "trial_split"] else self.dataset_params[
                                  "original_folder_trial_split"]),
            filter=False,
            median_filter=False,
            median_filter_size=(1, 3, 3),
            z_score=False,
            slice_stack=self.dataset_params[
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
                "crop_y"], load_into_mem=self.load_into_mem,
            trial_split=self.dataset_params["trial_split"],
            trial_split_length=self.dataset_params["trial_length"], trial_num=trial_num,
            zarr_path=False if not self.dataset_params["single_file_mode"] else
            os.path.join(self.save_dir_path, "temp_files/dataset.zarr")
        )
        self.total_size = total_size
        return stack

    @delayed
    def load_trial_filter_step(self, trial_num, dataset=False, loaded_num=False):
        """
        Delayed function step that applies filter to a single trial
        Parameters
        ----------
        trial_num : int
            trial num to load in trials_all
        dataset : ndarray
            Dataset to process if false load dataset

        Returns
        -------
        Filtered trial as a np.ndarray
        """
        dataset = dataset if type(
            dataset) != bool else self.load_trial_dataset_step(
            trial_num).compute()
        # dataset =dataset.astype(np.float32)
        if self.auto_crop:
            crop_y_bools = (dataset <= dataset.min()).all(1).any(0)
            y_iter_1 = 0
            while True:
                if crop_y_bools[y_iter_1]:
                    y_iter_1 += 1
                else:
                    break
            y_iter_2 = 0
            while True:
                if crop_y_bools[-(y_iter_2 + 1)]:
                    y_iter_2 += 1
                else:
                    break

            crop_x_bools = (dataset <= dataset.min()).all(2).any(0)
            x_iter_1 = 0
            while True:
                if crop_x_bools[x_iter_1]:
                    x_iter_1 += 1
                else:
                    break
            x_iter_2 = 0
            while True:
                if crop_x_bools[-(x_iter_2 + 1)]:
                    x_iter_2 += 1
                else:
                    break

            self.suggested_crops[trial_num] = [[x_iter_1, x_iter_2],
                                               [y_iter_1, y_iter_2]]

        cur_stack = filter_stack(

            stack=dataset,
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
        del dataset
        if not self.load_into_mem:

            z1 = zarr.open(os.path.join(self.save_dir_path,
                                        'temp_files/%s.zarr' % self.trials_all[
                                            trial_num]), mode='w',
                           shape=cur_stack.shape,
                           chunks=(32, 128, 128), dtype=np.float32)
            z1[:] = cur_stack
            if type(loaded_num) != bool:
                self.mean_images[loaded_num] = np.mean(cur_stack, axis=0)
                self.max_images[loaded_num] = np.max(cur_stack, axis=0)
                # self.temporal_correlation_images[
                #     loaded_num] = calculate_temporal_correlation(cur_stack).compute()


        else:

            if type(loaded_num) != bool:
                self.mean_images[loaded_num] = np.mean(cur_stack, axis=0)
                self.max_images[loaded_num] = np.max(cur_stack, axis=0)
                # self.temporal_correlation_images[
                #     loaded_num] = calculate_temporal_correlation(
                #     cur_stack).compute()
        if self.filter_params["pca"] and type(loaded_num) != bool:
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
        with open(os.path.join(self.save_dir_path,
                               'temp_files/filter/%s' % self.trials_all[
                                   trial_num]), "w") as f:
            f.write("done")
        printProgressBarFilter(
            total_num_spatial_boxes=self.box_params["total_num_spatial_boxes"],
            total_num_time_steps=self.box_params["total_num_time_steps"],
            save_dir=self.save_dir_path, progress_signal=self.progress_signal)

        if self.load_into_mem:
            return cur_stack
        else:
            return z1

    def calculate_filters(self, progress_signal=None, auto_crop=False):
        """
        Applies filter to each trial, sets them to self.dataset_trials_filtered

        Returns
        -------
        A list of filtered trials
        """
        if not self.auto_crop:
            self.dataset_params["auto_crop"] = auto_crop
        if self.global_params["need_recalc_filter_params"] or self.global_params[
            "need_recalc_dataset_params"] or \
                not hasattr(self, "dataset_trials_filtered"):
            self.progress_signal = progress_signal
            print("Started Calculating Filters")

            self.update_trial_list()

            save_dir = self.save_dir_path
            if not os.path.isdir(os.path.join(save_dir, "temp_files/filter")):
                os.mkdir(os.path.join(save_dir, "temp_files/filter"))
            filelist = [f for f in
                        os.listdir(os.path.join(save_dir, "temp_files/filter"))]
            for f in filelist:
                os.remove(
                    os.path.join(os.path.join(save_dir, "temp_files/filter"), f))

            printProgressBarFilter(
                total_num_spatial_boxes=self.box_params["total_num_spatial_boxes"],
                total_num_time_steps=self.box_params["total_num_time_steps"],
                save_dir=self.save_dir_path, progress_signal=self.progress_signal)
            if self.auto_crop:
                self.suggested_crops = [[[0, 0], [0, 0]] for _ in self.trials_all]
            self.dataset_trials_filtered = [False] * len(self.trials_all)
            if self.filter_params["pca"]:
                self.pca_decomp = [False] * len(self._trials_loaded_indices)
            self.max_images = [False] * len(self._trials_loaded_indices)
            self.mean_images = [False] * len(self._trials_loaded_indices)

            self.temporal_correlation_images = [False] * len(
                self._trials_loaded_indices)
            if not self.load_into_mem:
                for num, trial_num in enumerate(self._trials_loaded_indices):
                    self.dataset_trials_filtered[
                        trial_num] = self.load_trial_filter_step(
                        trial_num, self.load_trial_dataset_step(trial_num),
                        loaded_num=num)
                    if num % 3 == 0:
                        self.dataset_trials_filtered = list(
                            dask.compute(*self.dataset_trials_filtered))

                self.dataset_trials_filtered = list(
                    dask.compute(*self.dataset_trials_filtered))
            else:
                for num, trial_num in enumerate(self._trials_loaded_indices):
                    self.dataset_trials_filtered[
                        trial_num] = self.load_trial_filter_step(
                        trial_num, self.load_trial_dataset_step(trial_num),
                        loaded_num=num)
                if num % 7 == 0:
                    self.dataset_trials_filtered = list(
                        dask.compute(*self.dataset_trials_filtered))
                self.dataset_trials_filtered = list(
                    dask.compute(*self.dataset_trials_filtered))
            self.shape = [
                self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[1],
                self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[2]]
            if self.dataset_params["crop_x"][1] == 0:
                self.dataset_params["crop_x"][1] = self.shape[0]
                self.dataset_params["crop_y"][1] = self.shape[1]
            if self.auto_crop:
                crop = [[max([x[0][0] for x in self.suggested_crops]),
                         -max([x[0][1] for x in self.suggested_crops])],
                        [max([x[1][0] for x in self.suggested_crops]),
                         -max([x[1][1] for x in self.suggested_crops])]]

                self.dataset_params["crop_x"][0] = 0 + \
                                                   crop[0][0]
                self.dataset_params["crop_x"][1] = self.total_size[0] + \
                                                   crop[0][1]
                self.dataset_params["crop_y"][0] = 0 + \
                                                   crop[1][0]
                self.dataset_params["crop_y"][1] = self.total_size[1] + \
                                                   crop[1][1]
                if crop[0][1] == 0:
                    crop[0][1] = self.total_size[0]
                if crop[1][1] == 0:
                    crop[1][1] = self.total_size[1]
                for trial_num in self._trials_loaded_indices:
                    self.dataset_trials_filtered[trial_num] = \
                        self.dataset_trials_filtered[trial_num][
                        :, crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
                    self.mean_images[trial_num] = self.mean_images[trial_num][
                                                  crop[0][0]:crop[0][1],
                                                  crop[1][0]:crop[1][1]]
                    self.max_images[trial_num] = self.max_images[trial_num][
                                                 crop[0][0]:crop[0][1],
                                                 crop[1][0]:crop[1][1]]
                    # self.temporal_correlation_images[trial_num] = \
                    #     self.temporal_correlation_images[trial_num][
                    #     crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
                    if self.filter_params["pca"]:
                        self.pca_decomp[trial_num] = self.pca_decomp[trial_num][
                                                     :, crop[0][0]:crop[0][1],
                                                     crop[1][0]:crop[1][1]]
                self.dataset_params["crop_stack"] = True
                self.dataset_params["auto_crop"] = False
                self.shape = [
                    self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[
                        1],
                    self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[
                        2]]

            mean_image_stack = np.dstack(self.mean_images)
            with open(os.path.join(self.save_dir_path,
                                   'temp_files/filter/mean'), "w") as f:
                f.write("done")
            printProgressBarFilter(
                total_num_spatial_boxes=self.box_params["total_num_spatial_boxes"],
                total_num_time_steps=self.box_params["total_num_time_steps"],
                save_dir=self.save_dir_path, progress_signal=self.progress_signal)

            max_image_stack = np.dstack(self.max_images)
            with open(os.path.join(self.save_dir_path,
                                   'temp_files/filter/max'), "w") as f:
                f.write("done")
            mean_image_zarr = zarr.open(os.path.join(self.save_dir_path,
                                                     'temp_files/mean.zarr'), mode='w',
                                        shape=mean_image_stack.shape,
                                        chunks=(40, 40, 1))
            mean_image_zarr[:] = mean_image_stack

            max_image_zarr = zarr.open(os.path.join(self.save_dir_path,
                                                    'temp_files/max.zarr'), mode='w',
                                       shape=mean_image_stack.shape,
                                       chunks=(40, 40, 1))
            max_image_zarr[:] = max_image_stack
            printProgressBarFilter(
                total_num_spatial_boxes=self.box_params["total_num_spatial_boxes"],
                total_num_time_steps=self.box_params["total_num_time_steps"],
                save_dir=self.save_dir_path, progress_signal=self.progress_signal)

            # self.mean_images = [np.mean(x[:], axis=0) for x in
            #                     self.dataset_trials_filtered_loaded]
            #
            # self.max_images = [np.max(x[:], axis=0) for x in
            #                    self.dataset_trials_filtered_loaded]

            # self.temporal_correlation_image = calculate_temporal_correlation(self.dataset_filtered)
            self.global_params["need_recalc_filter_params"] = False
            self.global_params["need_recalc_dataset_params"] = False

            # self.global_params["need_recalc_box_params"] = True
            self.save_new_param_json()
            self.delete_roi_vars()
            # self.global_params["need_recalc_eigen_params"] = True
            if self.filter_params["pca"]:
                with open(os.path.join(self.save_dir_path,
                                       'pca_shape.text'), "w") as f:
                    f.write("shapes")
                    f.write(str([x.shape for x in self.pca_decomp if type(x)!= bool]))
        return self.dataset_trials_filtered

    @property
    def real_trials(self):
        return not (self.dataset_params["single_file_mode"] or self.dataset_params[
            "trial_split"] or len(self.trials_loaded) == 1)

    def load_data(self):
        self.update_trial_list()
        self.dataset_trials_filtered = [False] * len(self.trials_all)
        if self.filter_params["pca"]:
            self.pca_decomp = [False] * len(self.trials_loaded_time_trace_indices)
        for trial_num in self._trials_loaded_indices:
            self.dataset_trials_filtered[trial_num] = zarr.open(
                os.path.join(self.save_dir_path,
                             'temp_files/%s.zarr' % self.trials_all[trial_num]),
                mode="r")
            if self.filter_params["pca"]:
                self.pca_decomp[trial_num] = zarr.open(
                    os.path.join(self.save_dir_path,
                                 'temp_files/%s_pca.zarr' % self.trials_all[trial_num]),
                    mode="r")
        max_image_zarr = zarr.open(
            os.path.join(self.save_dir_path,
                         'temp_files/max.zarr'),
            mode="r")
        self.max_images = [False] * len(self.trials_loaded_time_trace_indices)
        mean_image_zarr = zarr.open(
            os.path.join(self.save_dir_path,
                         'temp_files/mean.zarr'),
            mode="r")
        self.mean_images = [False] * len(self.trials_loaded_time_trace_indices)
        for x in range(len(self._trials_loaded_indices)):
            self.max_images[x] = max_image_zarr[:, :, x]
            self.mean_images[x] = mean_image_zarr[:, :, x]
        self.shape = [
            self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[1],
            self.dataset_trials_filtered[self._trials_loaded_indices[0]].shape[2]]
        if self.dataset_params["crop_x"][1] == 0:
            self.dataset_params["crop_x"][1] = self.shape[0]
            self.dataset_params["crop_y"][1] = self.shape[1]
        # self.mean_images = [np.mean(x[:], axis=0) for x in
        #                     self.dataset_trials_filtered_loaded]
        #
        # self.max_images = [np.max(x[:], axis=0) for x in
        #                    self.dataset_trials_filtered_loaded]
        self.global_params["need_recalc_filter_params"] = False
        self.global_params["need_recalc_dataset_params"] = False

    def update_trial_list(self):
        if self.dataset_params["trial_split"] and self.dataset_params[
            "original_folder_trial_split"] != "":
            if self.dataset_params["single_file_mode"]:

                z1 = zarr.open(
                    os.path.join(self.save_dir_path, "temp_files/dataset.zarr"),
                    mode="r")
                self.trials_loaded = [str(x) for x in range(ceil(z1.shape[0] /
                                                                 self.dataset_params[
                                                                     "trial_length"]))]
            else:
                self.trials_loaded = [str(x) for x in range(ceil(len(os.listdir(
                    os.path.join(self.dataset_params["dataset_folder_path"],
                                 self.dataset_params["original_folder_trial_split"]))) /
                                                                 self.dataset_params[
                                                                     "trial_length"]))]
            self.trials_all = self.trials_loaded.copy()
            self.box_params["total_num_time_steps"] = len(self.trials_loaded)
        elif self.dataset_params["original_folder_trial_split"] != "":
            self.trials_loaded = [
                self.dataset_params["original_folder_trial_split"]]
            self.trials_all = [self.dataset_params["original_folder_trial_split"]]
            self.box_params["total_num_time_steps"] = len(self.trials_loaded)
        if self.dataset_params["original_folder_trial_split"] != "":
            self._trials_loaded_indices = [num for num, x in
                                           enumerate(self.trials_all)
                                           if x in self.trials_loaded]
            self.trials_loaded_time_trace_indices = [num for num, x in
                                                     enumerate(self.trials_all)
                                                     if x in self.trials_loaded]

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
                self.rois = process_data(num_threads=self.global_params[
                    "num_threads"], test_images=False, test_output_dir="",
                                         save_dir=self.save_dir_path,
                                         save_intermediate_steps=self.global_params[
                                             "save_intermediate_steps"],
                                         image_data=self.dataset_trials_filtered_loaded,
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
                                         roi_eccentricity_limit=self.roi_extraction_params["roi_eccentricity_limit"],
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

        try:
            self.eigen_norm_image = np.asarray(Image.open(
                os.path.join(self.save_dir_path,
                             "embedding_norm_images/embedding_norm_image.png")))
        except:
            print("Can't generate eigen Norm image please try again")

    def calculate_time_traces(self, report_progress=None):
        """
        Calculates the time traces for every roi in self.rois
        """
        self.neuropil_pixels = calculate_nueropil(image_shape=self.shape,
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
                            list(DataHandler.time_trace_possibilities_functions)}
        for x in self.time_traces.keys():
            for _ in range(len(self.rois)):
                self.time_traces[x].append([])
                for _ in range(len(self.trials_all)):
                    self.time_traces[x][-1].append(False)
        calc_list = []
        roi_time_traces_by_pixel = []
        for _ in range(len(self.rois)):
            roi_time_traces_by_pixel.append([])
            for _ in range(len(self.trials_all)):
                roi_time_traces_by_pixel[-1].append(False)
        roi_neuropil_traces_by_pixel = []
        for _ in range(len(self.rois)):
            roi_neuropil_traces_by_pixel.append([])
            for _ in range(len(self.trials_all)):
                roi_neuropil_traces_by_pixel[-1].append(False)
        for trial_num in self.trials_loaded_time_trace_indices:
            data = self.load_trial_dataset_step(trial_num).compute()
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
        if not self.real_trials:
            for roi_counter, roi_data, neuropil_data in zip(range(len(self.rois)),
                                                            roi_time_traces_by_pixel,
                                                            roi_neuropil_traces_by_pixel):
                roi_data_combined = np.hstack(
                    [roi_data[x] for x in self.trials_loaded_time_trace_indices])
                neuropil_data_combined = np.hstack(
                    [neuropil_data[x] for x in self.trials_loaded_time_trace_indices])
                roi_data_denoised_combined = waveletDenoise(roi_data_combined)
                for key in self.time_traces.keys():
                    self.time_traces[key][roi_counter] = [
                        DataHandler.time_trace_possibilities_functions[key](
                            roi_data_combined, neuropil_data_combined,
                            roi_data_denoised_combined)]
                if report_progress is not None:
                    printProgressBar(
                        len(
                            self.trials_loaded_time_trace_indices) + roi_counter + 1,
                        total=len(
                            self.trials_loaded_time_trace_indices) + len(self.rois) + 2,
                        prefix="Time Trace Calculation Progress:",
                        suffix="Complete", progress_signal=report_progress)
        if self.real_trials:
            for roi_counter, roi_data, neuropil_data in zip(range(len(self.rois)),
                                                            roi_time_traces_by_pixel,
                                                            roi_neuropil_traces_by_pixel):
                roi_data_denoised = [waveletDenoise(x) if type(x) != bool else False for
                                     x in roi_data]
                for key in self.time_traces.keys():
                    for trial_num in self.trials_loaded_time_trace_indices:
                        self.time_traces[key][roi_counter][trial_num] = \
                            DataHandler.time_trace_possibilities_functions[key](
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
                                    print_info=False,roi_eccentricity_limit=1).compute()

            if len(roi) > 0 and box_point_1d in roi[0]:
                rois.append(box.redefine_spatial_cord_1d(roi).compute()[0])
        if len(rois) > 0:
            final_roi = reduce(combine_rois, rois)
        else:
            return []

        return final_roi

    def calculate_statistics(self):
        statistics = [False for x in range(len(self.rois))]
        A = np.zeros([self.edge_roi_image_flat.shape[0], len(self.rois)], dtype=int)
        for num, roi in enumerate(self.rois):
            A[roi, num] = 1
        rval, significant_samples = classify_components_ep(
            self.dataset_trials_filtered_loaded, A,
            np.vstack([self.get_time_trace(x + 1) for x in range(len(self.rois))]))
        pass

    def export(self, matlab=False):
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
        shape = self.edge_roi_image_flat.shape
        roi_image_blob = self.pixel_with_rois_color_flat
        roi_image_outline = np.hstack(
            [self.edge_roi_image_flat,
             np.zeros(shape),
             np.zeros(shape)])
        max_image = np.reshape(
            np.max([self.max_images[x] for x in self._trials_loaded_indices], axis=0),
            (-1, 1))
        roi_image_blob_w_background = roi_image_blob + np.hstack(
            [max_image, max_image, max_image]) / max_image.max() * 255
        roi_image_outline_w_background = roi_image_outline + np.hstack(
            [max_image, max_image, max_image]) / max_image.max() * 255
        self.save_image(roi_image_blob,
                        os.path.join(self.save_dir_path, "roi_blob.png"))
        self.save_image(roi_image_outline,
                        os.path.join(self.save_dir_path, "roi_outline.png"))
        self.save_image(roi_image_outline_w_background,
                        os.path.join(self.save_dir_path, "roi_outline_background.png"))
        self.save_image(roi_image_blob_w_background,
                    os.path.join(self.save_dir_path, "roi_blob_background.png"))
        self.save_image(np.hstack(
            [max_image, max_image, max_image]),
            os.path.join(self.save_dir_path, "max.png"))
        if True:
            test = {x[:31].replace(" ", "_"): np.vstack(self.time_traces[x]) for x in
                    self.time_traces.keys()}

            savemat(os.path.join(self.save_dir_path, "time_traces.mat"), {"data": test},
                    appendmat=True)
            savemat(os.path.join(self.save_dir_path, "rois.mat"), {"data": self.rois},
                    appendmat=True)



    def save_image(self, image, path):
        img = Image.fromarray(
            (np.reshape(image - image.min(),
                        (self.shape[0], self.shape[1], 3)) / (
                     image.max() - image.min()) * 255).astype("uint8"))
        img.save(path)

    def delete_roi_vars(self):
        self.rois = []
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
