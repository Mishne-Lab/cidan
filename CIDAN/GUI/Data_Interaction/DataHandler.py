import json
import logging

import dask
import numpy as np
from PIL import Image
from dask import delayed
from skimage import feature

from CIDAN.LSSC.functions.data_manipulation import load_filter_tif_stack, filter_stack, \
    reshape_to_2d_over_time, pixel_num_to_2d_cord
from CIDAN.LSSC.functions.pickle_funcs import *
from CIDAN.LSSC.process_data import process_data

logger1 = logging.getLogger("CIDAN.DataHandler")


class DataHandler:
    _global_params_default = {
        "save_intermediate_steps": True,
        "need_recalc_dataset_params": True,
        "need_recalc_filter_params": True,
        "need_recalc_box_params": True,
        "need_recalc_eigen_params": True,
        "need_recalc_roi_extraction_params": True,
        "num_threads": 1
    }

    _dataset_params_default = {
        "dataset_path": "",
        "trials_loaded": [],
        "trials_all": [],
        "slice_stack": False,
        "slice_every": 3,
        "slice_start": 0
    }

    _filter_params_default = {
        "median_filter": False,
        "median_filter_size": 3,
        "z_score": False

    }
    _box_params_default = {
        "total_num_time_steps": 1,
        "total_num_spatial_boxes": 1,
        "spatial_overlap": 30
    }
    _eigen_params_default = {
        "eigen_vectors_already_generated": False,
        "num_eig": 50,
        "normalize_w_k": 16,
        "metric": "l2",
        "knn": 35,
        "accuracy": 50,
        "connections": 51,

    }
    _roi_extraction_params_default = {
        "elbow_threshold_method": True,
        "elbow_threshold_value": 1,
        "eigen_threshold_method": True,
        "eigen_threshold_value": .1,
        "num_eigen_vector_select": 7,
        "merge_temporal_coef": .95,
        "roi_size_min": 30,
        "roi_size_max": 600,
        "merge": True,
        "num_rois": 25,
        "fill_holes": True,
        "refinement": True,
        "max_iter": 1000,
    }

    def __init__(self, data_path, save_dir_path, save_dir_already_created, trials=[]):
        # TODO add loaded trials and all trials parameter here
        # TODO make sure if trial list includes files that aren't valid it works
        self.color_list = [(218, 67, 34),
                           (132, 249, 22), (22, 249, 140), (22, 245, 249),
                           (22, 132, 249), (224, 22, 249), (249, 22, 160)]

        self.save_dir_path = save_dir_path
        self.rois_loaded = False
        if save_dir_already_created:
            valid = self.load_param_json()
            self._trials_loaded_indices = [num for num, x in enumerate(self.trials_all)
                                           if x in self.trials_loaded]
            self.trials_loaded_time_trace_indices = [num for num, x in
                                                     enumerate(self.trials_all)
                                                     if x in self.trials_loaded]
            self.calculate_filters()

            if self.rois_exist:
                self.load_rois()
                self.calculate_time_traces()
                self.rois_loaded = True
            if not valid:
                raise FileNotFoundError("Save directory not valid")
        else:
            self.global_params = DataHandler._global_params_default.copy()

            self.dataset_params = DataHandler._dataset_params_default.copy()
            self.dataset_params["dataset_path"] = data_path
            self.dataset_params["trials_loaded"] = trials
            self.trials_loaded = trials

            self.trials_all = sorted(os.listdir(data_path))
            self.trials_all = [x for x in self.trials_all if ".tif" in x]
            self.dataset_params["trials_all"] = self.trials_all

            self.filter_params = DataHandler._filter_params_default.copy()
            self.box_params = DataHandler._box_params_default.copy()
            self.box_params["total_num_time_steps"] = len(trials)
            self.eigen_params = DataHandler._eigen_params_default.copy()
            self.roi_extraction_params = DataHandler._roi_extraction_params_default.copy()
            valid = self.create_new_save_dir()
            if not valid:
                raise FileNotFoundError("Please chose an empty directory for your " +
                                        "save directory")
            self.time_traces = []
            self._trials_loaded_indices = [num for num, x in enumerate(self.trials_all)
                                           if x in self.trials_loaded]
            self.trials_loaded_time_trace_indices = [num for num, x in
                                                     enumerate(self.trials_all)
                                                     if x in self.trials_loaded]

    def __del__(self):
        for x in self.__dict__.items():
            self.__dict__[x] = None

    @property
    def dataset_trials_filtered_loaded(self):
        return [self.dataset_trials_filtered[x] for x in self._trials_loaded_indices]

    @property
    def dataset_trials_loaded(self):
        return [self.dataset_trials[x] for x in self._trials_loaded_indices]
    @property
    def param_path(self):
        return os.path.join(self.save_dir_path, "parameters.json")

    @property
    def eigen_vectors_exist(self):
        eigen_dir = os.path.join(self.save_dir_path, "eigen_vectors")

        file_names = [
            "eigen_vectors_box_{}_{}.pickle".format(spatial_box_num, time_box_num)
            for spatial_box_num in range(self.box_params["total_num_spatial_boxes"])
            for time_box_num in range(self.box_params["total_num_time_steps"])]
        return all(pickle_exist(x, output_directory=eigen_dir) for x in file_names)

    @property
    def rois_exist(self):
        return pickle_exist("rois", output_directory=self.save_dir_path)

    def load_rois(self):
        if pickle_exist("rois", output_directory=self.save_dir_path):
            self.clusters = pickle_load("rois", output_directory=self.save_dir_path)
            self.gen_roi_display_variables()

    def save_rois(self, rois):
        if os.path.isdir(self.save_dir_path):
            pickle_save(rois, "rois", output_directory=self.save_dir_path)

    def load_param_json(self):

        try:
            with open(self.param_path, "r") as f:
                all_params = json.loads(f.read())
            self.global_params = all_params["global_params"]
            self.dataset_params = all_params["dataset_params"]
            self.filter_params = all_params["filter_params"]
            self.box_params = all_params["box_params"]
            self.eigen_params = all_params["eigen_params"]
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
        try:
            with open(self.param_path, "w") as f:
                all_params = {
                    "global_params": self.global_params,
                    "dataset_params": self.dataset_params,
                    "filter_params": self.filter_params,
                    "box_params": self.box_params,
                    "eigen_params": self.eigen_params,
                    "roi_extraction_params": self.roi_extraction_params
                }
                f.truncate(0)
                f.write(json.dumps(all_params))
        except:
            raise FileNotFoundError("Error saving parameters, please restart software")
        pass

    def create_new_save_dir(self):
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
            return True
        except:
            raise FileNotFoundError("Couldn't create folder please try again")

    def change_global_param(self, param_name, new_value):
        if param_name in self.global_params:
            self.global_params[param_name] = new_value
            self.save_new_param_json()
            return True
        else:
            return False

    def change_dataset_param(self, param_name, new_value):
        if param_name in self.dataset_params:
            self.dataset_params[param_name] = new_value
            self.global_params["need_recalc_dataset_params"] = True
            self.save_new_param_json()
            return True
        else:
            return False

    def change_filter_param(self, param_name, new_value):
        if param_name in self.filter_params:
            self.filter_params[param_name] = new_value
            self.global_params["need_recalc_filter_params"] = True
            self.save_new_param_json()
            return True
        else:
            return False

    def change_box_param(self, param_name, new_value):
        if param_name in self.box_params:
            # if param_name == "total_num_spatial_boxes":
            #     assert (int(new_value**.5))**2 == new_value, "Please make sure Number of Spatial Boxes is a square number"
            self.box_params[param_name] = new_value
            self.global_params["need_recalc_box_params"] = True
            self.global_params["need_recalc_eigen_params"] = True
            self.global_params["need_recalc_roi_extraction_params"] = True
            self.save_new_param_json()
            return True
        else:
            return False

    def change_eigen_param(self, param_name, new_value):
        if param_name in self.eigen_params:
            self.eigen_params[param_name] = new_value
            self.global_params["need_recalc_eigen_params"] = True
            self.save_new_param_json()
            return True
        else:
            return False

    def change_roi_extraction_param(self, param_name, new_value):
        if param_name in self.roi_extraction_params:
            self.roi_extraction_params[param_name] = new_value
            self.global_params["need_recalc_roi_extraction_params"] = True
            self.save_new_param_json()
            return True
        else:
            return False

    def calculate_dataset(self) -> np.ndarray:
        """Loads the dataset

        Returns
        -------
        """
        # TODO make it so it does't load the dataset every time

        self.dataset_trials = [False] * len(self.trials_all)
        for trial_num in self._trials_loaded_indices:
            self.dataset_trials[trial_num] = self.load_trial_dataset_step(trial_num)
        self.dataset_trials = dask.compute(*self.dataset_trials)
        self.global_params["need_recalc_dataset_params"] = False
        self.shape = [self.dataset_trials_loaded[0].shape[1],
                      self.dataset_trials_loaded[0].shape[2]]

        print("Finished Calculating Dataset")

    @delayed
    def load_trial_dataset_step(self, trial_num):
        return load_filter_tif_stack(
            path=os.path.join(self.dataset_params["dataset_path"],
                              self.trials_all[trial_num]),
            filter=False,
            median_filter=False,
            median_filter_size=(1, 3, 3),
            z_score=False,
            slice_stack=self.dataset_params[
                "slice_stack"],
            slice_start=self.dataset_params[
                "slice_start"],

            slice_every=self.dataset_params[
                "slice_every"])

    @delayed
    def load_trial_filter_step(self, trial_num):

        return filter_stack(
            stack=self.dataset_trials[trial_num] if type(self.dataset_trials[
                                                             trial_num]) != bool else self.load_trial_dataset_step(
                trial_num).compute(),
            median_filter_size=(1,
                                self.filter_params[
                                    "median_filter_size"],
                                self.filter_params[
                                    "median_filter_size"]),
            median_filter=self.filter_params[
                "median_filter"],
            z_score=self.filter_params["z_score"])

    def calculate_filters(self):

        if self.global_params["need_recalc_filter_params"] or self.global_params[
            "need_recalc_dataset_params"] or \
                not hasattr(self, "dataset_trials_filtered"):
            self.calculate_dataset()
            self.dataset_trials_filtered = [False] * len(self.trials_all)
            for trial_num in self._trials_loaded_indices:
                self.dataset_trials_filtered[trial_num] = self.load_trial_filter_step(
                    trial_num)
            self.dataset_trials_filtered = dask.compute(*self.dataset_trials_filtered)
            self.mean_image = np.mean(np.dstack(
                [np.mean(x, axis=0) for x in self.dataset_trials_filtered_loaded]),
                                      axis=2)
            self.max_image = np.max(np.dstack(
                [np.max(x, axis=0) for x in self.dataset_trials_filtered_loaded]),
                                    axis=2)

            # self.temporal_correlation_image = calculate_temporal_correlation(self.dataset_filtered)
            self.global_params["need_recalc_filter_params"] = False
            self.global_params["need_recalc_box_params"] = True
        return self.dataset_trials_filtered

    def calculate_roi_extraction(self):
        if self.global_params["need_recalc_eigen_params"] or self.global_params[
            "need_recalc_roi_extraction_params"] or self.global_params[
            "need_recalc_box_parmas"] or self.global_params[
            "need_recalc_dataset_params"] or \
                self.global_params["need_recalc_filter_params"]:
            assert (int(
                self.box_params[
                    "total_num_spatial_boxes"] ** .5)) ** 2 == self.box_params[
                       "total_num_spatial_boxes"], "Please make sure Number of Spatial Boxes is a square number"
            try:
                self.calculate_filters()
                self.clusters = process_data(num_threads=self.global_params[
                    "num_threads"], test_images=False, test_output_dir="",
                                             save_dir=self.save_dir_path,
                                             save_intermediate_steps=self.global_params[
                                                 "save_intermediate_steps"],
                                             image_data=self.dataset_trials_filtered_loaded,
                                             eigen_vectors_already_generated=(not
                                                                              self.global_params[
                                                                                  "need_recalc_eigen_params"]) and
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
                                                 "roi_size_max"])

                self.global_params["need_recalc_eigen_params"] = False
                self.save_rois(self.clusters)
                print("Calculating Time Traces:")
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
        cluster_list_2d_cord = [
            pixel_num_to_2d_cord(x, volume_shape=self.shape) for x in
            self.clusters]
        self.cluster_max_cord_list = [np.max(x, axis=1) for x in cluster_list_2d_cord]
        self.cluster_min_cord_list = [np.min(x, axis=1) for x in cluster_list_2d_cord]
        self.pixel_with_rois_flat = np.zeros(
            [self.shape[0] * self.shape[1]])
        self.pixel_with_rois_color_flat = np.zeros(
            [self.shape[0] * self.shape[1], 3])
        for num, cluster in enumerate(self.clusters):
            cur_color = self.color_list[num % len(self.color_list)]
            self.pixel_with_rois_flat[cluster] = num + 1
            self.pixel_with_rois_color_flat[cluster] = cur_color
        edge_roi_image = feature.canny(np.reshape(self.pixel_with_rois_flat,
                                                  [self.shape[0],
                                                   self.shape[1]]))
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

    def calculate_time_traces(self):
        self.time_traces = []
        for _ in range(len(self.clusters)):
            self.time_traces.append([])
            for _ in range(len(self.trials_all)):
                self.time_traces[-1].append(False)
        for trial_num in self.trials_loaded_time_trace_indices:
            for cluster in range(len(self.clusters)):
                self.calculate_time_trace(cluster + 1, trial_num)

        if os.path.isdir(self.save_dir_path):
            pickle_save(self.time_traces, "time_traces",
                        output_directory=self.save_dir_path)
        self.rois_loaded = True

    def calculate_time_trace(self, roi_num, trial_num=None):
        """
        Calculates a time trace for a certain ROI and save to time trace list
        Parameters
        ----------
        roi_num
            roi to calculate for this starts at [1..number of rois]
        trial_num
            indice of trial in trials_all starts at [0, number of trials-1] if
            none then calculates for all trials

        Returns
        -------

        """
        # TODO make so it can also calculate for a specific trial, so it would be a 2d array
        trial_nums = [trial_num]
        if trial_num == None:
            trial_nums = self.trials_loaded_time_trace_indices
        for trial_num in trial_nums:
            cluster = self.clusters[roi_num - 1]
            if type(self.dataset_trials_filtered[trial_num]) == bool:
                self.load_trial_filter_step(trial_num)
            data_2d = reshape_to_2d_over_time(self.dataset_trials_filtered[trial_num])
            time_trace = np.mean(data_2d[cluster], axis=0)
            self.time_traces[roi_num - 1][trial_num] = time_trace
        if os.path.isdir(self.save_dir_path):
            pickle_save(self.time_traces, "time_traces",
                        output_directory=self.save_dir_path)

    def get_time_trace(self, num):
        num = num - 1
        output = np.ndarray([])
        for trial_num in self.trials_loaded_time_trace_indices:
            if type(self.time_traces[num][trial_num]) == bool:
                self.calculate_time_trace(num + 1, trial_num)
            output = np.hstack([output, self.time_traces[num][trial_num]])

        return output

    def update_selected_trials(self, selected_trials):
        """
        Updates the loaded trials so don't have unnecessary loaded trials
        Parameters
        ----------
        selected_trials
            trials to be loaded

        Returns
        -------

        """
        new_selected = [x for x, y in enumerate(self.trials_all) if
                        y in selected_trials]
        for trial_num, _ in enumerate(self.trials_all):
            if trial_num in new_selected and trial_num not in self.trials_loaded_time_trace_indices:
                self.dataset_trials_filtered = self.load_trial_filter_step(
                    trial_num).compute()
                self.trials_loaded_time_trace_indices.append(trial_num)
            if trial_num not in new_selected and trial_num in self.trials_loaded_time_trace_indices \
                    and trial_num not in self._trials_loaded_indices:
                self.dataset_trials_filtered[trial_num] = False

            if trial_num not in new_selected and trial_num in self.trials_loaded_time_trace_indices:
                self.trials_loaded_time_trace_indices.remove(trial_num)
