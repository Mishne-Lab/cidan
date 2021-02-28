import json
import logging

from cidan.LSSC.functions.pickle_funcs import *

logger1 = logging.getLogger("cidan.DataHandler")


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
