import json
import os
from typing import Dict
import numpy as np
from LSSC.functions.data_manipulation import load_filter_tif_stack, filter_stack, \
    reshape_to_2d_over_time
from LSSC.process_data import process_data
class DataHandler:
    global_params_default = {
        "save_intermediate_steps": False,
        "need_recalc_dataset_params": True,
        "need_recalc_filter_params": True,
        "need_recalc_box_params": True,
        "need_recalc_eigen_params": True,
        "need_recalc_roi_extraction_params": True,
        "num_threads": 1
    }

    dataset_params_default = {
        "dataset_path": "",
        "slice_stack": False,
        "slice_every": 3,
        "slice_start": 0
    }

    filter_params_default = {
        "median_filter": False,
        "median_filter_size": 3,
        "z_score": False

    }
    box_params_default = {
        "total_num_time_steps": 1,
        "total_num_spatial_boxes": 4,
        "spatial_overlap": 15
    }
    eigen_params_default = {
        "eigen_vectors_already_generated": False,
        "num_eig": 24,
        "normalize_w_k": 2,
        "metric": "l2",
        "knn": 50,
        "accuracy": 50,
        "connections": 60,

    }
    roi_extraction_params_default = {
        "elbow_threshold_method": True,
        "elbow_threshold_value": 1.0,
        "eigen_threshold_method": True,
        "eigen_threshold_value": .5,
        "num_eigen_vector_select": 5,
        "merge_temporal_coef": .05,
        "roi_size_min": 30,
        "roi_size_max": 600,
        "merge": True,
        "num_rois": 25,
        "fill_holes": True,
        "refinement": True,
        "max_iter": 1000,
    }
    def __init__(self, data_path, save_dir_path, save_dir_already_created):
        # TODO Create two initialization methods one with default parameters and one
        #  with loading save dir parameters
        self.save_dir_path = save_dir_path
        if save_dir_already_created:
            valid = self.load_param_xml()
            if not valid:
                raise FileNotFoundError("Save directory not valid")
        else:
            self.global_params = DataHandler.global_params_default.copy()

            self.dataset_params = DataHandler.dataset_params_default.copy()
            self.dataset_params["dataset_path"] = data_path

            self.filter_params = DataHandler.filter_params_default.copy()
            self.box_params = DataHandler.box_params_default.copy()
            self.eigen_params = DataHandler.eigen_params_default.copy()
            self.roi_extraction_params = DataHandler.roi_extraction_params_default.copy()
            valid = self.create_new_save_dir()
            if not valid:
                raise FileNotFoundError("Please chose an empty directory for your " +
                                        "save directory")

    @property
    def param_path(self):
        return os.path.join(self.save_dir_path, "parameters.json")

    def load_param_json(self):

        try:
            with open(self.param_path, "r") as f:
                all_params = json.load(f.read())
            self.global_params = all_params["global_params"]
            self.dataset_params = all_params["dataset_params"]
            self.filter_params = all_params["filter_params"]
            self.box_params = all_params["box_params"]
            self.eigen_params = all_params["eigen_params"]
            self.roi_extraction_params = all_params["roi_extraction_params"]
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
            if param_name == "total_num_spatial_boxes":
                assert (int(new_value**.5))**2 == new_value, "Please make sure Number of Spatial Boxes is a square number"
            self.filter_params[param_name] = new_value
            self.global_params["need_recalc_box_params"] = True
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

    def calculate_dataset(self)->np.ndarray:
        """Loads the dataset

        Returns
        -------
        """
        if self.global_params["need_recalc_dataset_params"] or not hasattr(
                self,"dataset"):
            print("Starting Calculating Dataset")
            self.dataset = load_filter_tif_stack(path=self.dataset_params[
                "dataset_path"], filter=False,
                                                 median_filter=False,
                                                 median_filter_size=(1,3,3),
                                                 z_score=False, slice_stack=self.dataset_params[
                    "slice_stack"],
                                                 slice_start=self.dataset_params["slice_start"],

                                                 slice_every=self.dataset_params["slice_every"])
            self.global_params["need_recalc_dataset_params"] = False


            print("Finished Calculating Dataset")

        return self.dataset

    def calculate_filters(self):

        if self.global_params["need_recalc_filter_params"] or self.global_params["need_recalc_dataset_params"] or \
                not hasattr(self,"dataset_filtered"):
            self.dataset_filtered = filter_stack(stack=self.calculate_dataset(),
                                                 median_filter_size=(1, self.filter_params["median_filter_size"], self.filter_params["median_filter_size"]),
                                                 median_filter=self.filter_params[
                                                     "median_filter"],
                                                 z_score=self.filter_params["z_score"])
            self.global_params["need_recalc_filter_params"] = False
        return self.dataset_filtered






    def calculate_roi_extraction(self):
        if self.global_params["need_recalc_eigen_params"] or self.global_params[
            "need_recalc_roi_extraction_params"] or self.global_params[
            "need_recalc_box_parmas"] or self.global_params["need_recalc_dataset_params"] or \
                self.global_params["need_recalc_filter_params"]:
            self.clusters = process_data(num_threads=self.global_params[
                "num_threads"], test_images=False, test_output_dir="",
                                         save_dir=self.save_dir_path,
                                         save_intermediate_steps=self.global_params[
                     "save_intermediate_steps"],
                                         load_data=False, data_path="",
                                         image_data=self.calculate_filters(),
                                         eigen_vectors_already_generated=(not self.global_params[
                     "need_recalc_eigen_params"])and self.global_params["save_intermediate_steps"],
                                         save_embedding_images=True,
                                         total_num_time_steps=self.box_params["total_num_time_steps"],
                                         total_num_spatial_boxes=self.box_params[
                                             "total_num_spatial_boxes"],
                                         spatial_overlap=self.box_params["spatial_overlap"]//2, filter= False,
                                         median_filter=False,
                                         median_filter_size=(1,3,3),
                                         z_score=False, slice_stack=False,
                                         slice_every=1, slice_start=0, metric=self.eigen_params["metric"],
                                         knn=self.eigen_params["knn"],
                                         accuracy=self.eigen_params["accuracy"],
                                         connections=self.eigen_params["connections"],
                                         normalize_w_k=self.eigen_params["normalize_w_k"],
                                         num_eig=self.eigen_params["num_eig"],
                                         merge=self.roi_extraction_params["merge"],
                                         num_rois=self.roi_extraction_params["num_rois"],
                                         refinement=self.roi_extraction_params["refinement"],
                                         num_eigen_vector_select=self.roi_extraction_params[
                              "num_eigen_vector_select"],
                                         max_iter=self.roi_extraction_params["max_iter"],
                                         roi_size_min=self.roi_extraction_params["roi_size_min"],
                                         fill_holes=self.roi_extraction_params[
                                                 "fill_holes"],

                                         elbow_threshold_method=self.roi_extraction_params[
                     "elbow_threshold_method"],
                                         elbow_threshold_value=self.roi_extraction_params["elbow_threshold_value"],
                                         eigen_threshold_method=self.roi_extraction_params[
                     "eigen_threshold_method"],
                                         eigen_threshold_value=self.roi_extraction_params[
                     "eigen_threshold_value"],
                                         merge_temporal_coef=self.roi_extraction_params["merge_temporal_coef"],
                                         roi_size_max=self.roi_extraction_params["roi_size_max"])
            self.global_params["need_recalc_eigen_params"] = False

            self.pixel_with_clusters_flat = np.zeros([self.dataset.shape[1]*self.dataset.shape[2]])
            self.pixel_with_clusters_color_flat = np.zeros([self.dataset.shape[1]*self.dataset.shape[2],3])
            color_list = [(218, 67, 34), (218, 134, 34 ),(245, 249, 22  ),(132, 249, 22),(22, 249, 140),(22, 245, 249),(22, 132, 249),(224, 22, 249),(249, 22, 160),(249, 160, 22)]
            for num, cluster in enumerate(self.clusters):
                cur_color = color_list[num%len(color_list)]
                self.pixel_with_clusters_flat[cluster] = num+1
                self.pixel_with_clusters_color_flat[cluster]=cur_color
            self.pixel_with_clusters_color = np.reshape(self.pixel_with_clusters_color_flat, [self.dataset.shape[1],self.dataset.shape[2],3])


