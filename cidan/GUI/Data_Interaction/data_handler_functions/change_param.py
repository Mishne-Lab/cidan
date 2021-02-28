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
