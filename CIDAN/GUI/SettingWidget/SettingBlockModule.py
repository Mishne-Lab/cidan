from qtpy.QtWidgets import *

from CIDAN.GUI.Inputs.BoolInput import BoolInput
from CIDAN.GUI.Inputs.FloatInput import FloatInput
from CIDAN.GUI.Inputs.IntInput import IntInput
from CIDAN.GUI.Inputs.IntRangeInput import IntRangeInput


class SettingBlockModule(QFrame):
    """
    A tab of settings in SettingsModule

    These are all the tabs in each of the settings area in the GUI, just specify each
    input in a list and it will add them together
    """
    def __init__(self, name, input_list):
        """
        Initializes the list
        Parameters
        ----------
        name : str
            Name of the list of modules
        input_list : List[Input]
            The list of inputs that are part of this section of settings
        """
        super().__init__()
        self.name = name
        self.input_list = input_list
        self.layout = QVBoxLayout()
        # Type options are int, float, bool, and str
        for input in input_list:
            self.layout.addWidget(input)

        self.setLayout(self.layout)


def filter_setting_block(main_widget):
    data_handler = main_widget.data_handler
    return SettingBlockModule("Filter Settings",
                              [BoolInput(display_name="Median filter:",
                                         program_name="median_filter",
                                         on_change_function=lambda x,
                                                                   y: data_handler.change_filter_param(
                                             x, y),
                                         default_val=data_handler.filter_params[
                                             "median_filter"],
                                         tool_tip="Whether to apply a median filter to each timestep",
                                         ),
                               IntInput(
                                   display_name="Median filter size:",
                                   program_name="median_filter_size",
                                   on_change_function=lambda x,
                                                             y: data_handler.change_filter_param(
                                       x, y),
                                   default_val=
                                   data_handler.filter_params[
                                       "median_filter_size"],
                                   tool_tip="The size of the median filter for each timestep",
                                   min=1, max=50, step=1),
                               BoolInput(
                                   display_name="Z-score:",
                                   program_name="z_score",
                                   on_change_function=lambda x,
                                                             y: data_handler.change_filter_param(
                                       x, y),
                                   default_val=
                                   data_handler.filter_params["z_score"],
                                   tool_tip="Whether to apply a z-score for each pixel across all the timesteps",
                               ),
                               BoolInput(display_name="Histogram Equalization Method:",
                                         program_name="hist_eq",
                                         on_change_function=lambda x,
                                                                   y: data_handler.change_filter_param(
                                             x, y),
                                         default_val=data_handler.filter_params[
                                             "hist_eq"],
                                         tool_tip="Whether to apply our histogram equalization method",
                                         ),
                               BoolInput(display_name="PCA:",
                                         program_name="pca",
                                         on_change_function=lambda x,
                                                                   y: data_handler.change_filter_param(
                                             x, y),
                                         default_val=data_handler.filter_params[
                                             "pca"],
                                         tool_tip="Whether to apply PCA decomposition to the dataset(PCA runs after other filters)",
                                         ),
                               FloatInput(
                                   display_name="PCA expression threshold",
                                   program_name="pca_threshold",
                                   on_change_function=lambda x,
                                                             y: data_handler.change_filter_param(
                                       x, y),
                                   default_val=
                                   data_handler.filter_params[
                                       "pca_threshold"],
                                   tool_tip="The percentage of the variance that the PCA will express",
                                   min=0.001, max=.999, step=.001),
                               ])


def dataset_setting_block(main_widget):
    data_handler = main_widget.data_handler
    return SettingBlockModule("Advanced Settings",
                              [BoolInput(display_name="Slice stack:",
                                         program_name="slice_stack",
                                         on_change_function=lambda x,
                                                                   y: data_handler.change_dataset_param(
                                             x, y),
                                         default_val=data_handler.dataset_params[
                                             "slice_stack"],
                                         tool_tip="Used to Select only "
                                                  "every x timestep",
                                         display_tool_tip=False),
                               IntInput(display_name="Slice every:",
                                        program_name="slice_every",
                                        on_change_function=lambda x,
                                                                  y: data_handler.change_dataset_param(
                                            x, y),
                                        default_val=data_handler.dataset_params[
                                            "slice_every"],
                                        tool_tip="Select every x timesteps",
                                        min=1, max=100, step=1),
                               IntInput(display_name="Slice start:",
                                        program_name="slice_start",
                                        on_change_function=lambda x,
                                                                  y: data_handler.change_dataset_param(
                                            x, y),
                                        tool_tip="Start selecting on x timestep",
                                        default_val=data_handler.dataset_params[
                                            "slice_start"],
                                        min=0, max=10000, step=1)
                               ])


def dataset_setting_block_crop(main_widget):
    data_handler = main_widget.data_handler
    return SettingBlockModule("Crop Settings",
                              [BoolInput(display_name="Crop stack:",
                                         program_name="crop_stack",
                                         on_change_function=lambda x,
                                                                   y: data_handler.change_dataset_param(
                                             x, y),
                                         default_val=data_handler.dataset_params[
                                             "crop_stack"],
                                         tool_tip="Used to crop image stack",
                                         display_tool_tip=False),
                               IntRangeInput(display_name="Crop X:",
                                             program_name="crop_y",
                                             on_change_function=lambda x,
                                                                       y: data_handler.change_dataset_param(
                                                 x, y),
                                             default_val=data_handler.dataset_params[
                                                 "crop_y"],
                                             tool_tip="Crop in x direction",
                                             min=0, max=10000, step=1),
                               IntRangeInput(display_name="Crop Y:",
                                             # different because the way we display images is weird x is first dim y is second dim
                                             program_name="crop_x",
                                             on_change_function=lambda x,
                                                                       y: data_handler.change_dataset_param(
                                                 x, y),
                                             default_val=data_handler.dataset_params[
                                                 "crop_x"],
                                             tool_tip="Crop in x direction",
                                             min=0, max=100000, step=1),
                               ] + ([BoolInput(
                                  display_name="Split into trials(recomended):",
                                  program_name="trial_split",
                                  on_change_function=lambda x,
                                                            y: data_handler.change_dataset_param(
                                      x, y),
                                  default_val=data_handler.dataset_params[
                                      "trial_split"],
                                  tool_tip="Splits the timesteps into separate trials better for processing "
                                           "every x timestep",
                                  display_tool_tip=False),
                                                                      IntInput(
                                                                          display_name="Trial Length",
                                                                          program_name="trial_length",
                                                                          on_change_function=lambda x,
                                                                             y: data_handler.change_dataset_param(
                                                       x, y),
                                                                          default_val=
                                                   data_handler.dataset_params[
                                                       "trial_length"],
                                                                          tool_tip="Length of each trial",
                                                                          display_tool_tip=False,
                                                                          min=50,
                                                                          max=2000,
                                                                          step=1)] if
                                                                  data_handler.dataset_params[
                                                                      "original_folder_trial_split"] != "" else []))


def multiprocessing_settings_block(main_widget):
    data_handler = main_widget.data_handler
    return SettingBlockModule("Multiprocessing Settings", [

        IntInput(
            display_name="Number of spatial boxes:",
            program_name="total_num_spatial_boxes",
            on_change_function=lambda x, y: data_handler.change_box_param(
                x, y),
            default_val=
            data_handler.box_params[
                "total_num_spatial_boxes"],
            tool_tip="Number of boxes to break the calculation into, please make sure it is a sqaure",
            min=1, max=10000, step=1)
        ,
        IntInput(
            display_name="Spatial overlap:",
            program_name="spatial_overlap",
            on_change_function=lambda x, y: data_handler.change_box_param(x, y),
            default_val=
            data_handler.box_params[
                "spatial_overlap"],
            tool_tip="Number of pixels to overlap each box",
            min=1, max=10000, step=1)])


def roi_extraction_settings_block(main_widget):
    # TODO fix input into spatial box number
    data_handler = main_widget.data_handler
    return SettingBlockModule("ROI Extraction Settings",
                              [
                                  IntInput(
                                      display_name="Max Number of ROIs per spatial box:",
                                      program_name="num_rois",
                                      on_change_function=lambda x,
                                                                y: data_handler.change_roi_extraction_param(
                                          x, y),
                                      default_val=data_handler.roi_extraction_params[
                                          "num_rois"],
                                      tool_tip="Max number of ROIs to select for each spatial box",
                                      min=0, max=10000, step=1),
                                  IntInput(
                                      display_name="ROI size minimum:",
                                      program_name="roi_size_min",
                                      on_change_function=lambda x,
                                                                y: data_handler.change_roi_extraction_param(
                                          x, y),
                                      default_val=
                                      data_handler.roi_extraction_params[
                                          "roi_size_min"],
                                      tool_tip="Minimum size in pixels for a region of interest",
                                      min=1, max=10000, step=1),
                                  IntInput(
                                      display_name="ROI size maximum:",
                                      program_name="roi_size_max",
                                      on_change_function=lambda x,
                                                                y: data_handler.change_roi_extraction_param(
                                          x, y),
                                      default_val=
                                      data_handler.roi_extraction_params[
                                          "roi_size_max"],
                                      tool_tip="Maximum size in pixels for a region of interest",
                                      min=1, max=1000000, step=1),
                                  # BoolInput(
                                  #     display_name="Run refinement step:",
                                  #     program_name="refinement",
                                  #     on_change_function=lambda x,
                                  #                               y: data_handler.change_roi_extraction_param(
                                  #         x, y),
                                  #     default_val=
                                  #     data_handler.roi_extraction_params[
                                  #         "refinement"],
                                  #     tool_tip="Whether to run the refinement step, greatly improves results"),
                                  FloatInput(
                                      display_name="Merge temporal coefficient:",
                                      program_name="merge_temporal_coef",
                                      on_change_function=lambda x,
                                                                y: data_handler.change_roi_extraction_param(
                                          x, y),
                                      default_val=
                                      data_handler.roi_extraction_params[
                                          "merge_temporal_coef"],
                                      tool_tip="The coefficient that determines if two overlapping regions are merged based on their temporal correlation",
                                      min=0, max=1, step=.01)
                                  #     BoolInput(
                                  #         display_name="Fill holes:",
                                  #         program_name="fill_holes",
                                  #         on_change_function=lambda x,
                                  #                                   y: data_handler.change_roi_extraction_param(
                                  #             x, y),
                                  #         default_val=
                                  #         data_handler.roi_extraction_params[
                                  #             "fill_holes"],
                                  #         tool_tip="Whether to fill holes in each roi"),
                                  #
                              ]
                              )


def roi_advanced_settings_block(main_widget):
    # TODO fix input into spatial box number
    data_handler = main_widget.data_handler
    return SettingBlockModule("Advanced Settings",
                              [IntInput(display_name="Number of eigen vectors:",
                                        program_name="num_eig",
                                        on_change_function=lambda x,
                                                                  y: data_handler.change_eigen_param(
                                            x, y),
                                        default_val=data_handler.eigen_params[
                                            "num_eig"],
                                        tool_tip="Number of eigen vectors to generate",
                                        min=1, max=10000, step=1),
                               FloatInput(display_name="Eigen threshold value:",
                                          program_name="eigen_threshold_value",
                                          on_change_function=lambda x,
                                                                    y: data_handler.change_roi_extraction_param(
                                              x, y),
                                          default_val=
                                          data_handler.roi_extraction_params[
                                              "eigen_threshold_value"],
                                          tool_tip="Number of eigen vectors to select at each point",
                                          min=0, max=1, step=.01),
                               FloatInput(display_name="Elbow threshold value:",
                                          program_name="elbow_threshold_value",
                                          on_change_function=lambda x,
                                                                    y: data_handler.change_roi_extraction_param(
                                              x, y),
                                          default_val=
                                          data_handler.roi_extraction_params[
                                              "elbow_threshold_value"],
                                          tool_tip="Number of eigen vectors to select at each point",
                                          min=0, max=1.5, step=.01),
                               BoolInput(
                                   display_name="Merge ROIs:",
                                   program_name="merge",
                                   on_change_function=lambda x,
                                                             y: data_handler.change_roi_extraction_param(
                                       x, y),
                                   default_val=
                                   data_handler.roi_extraction_params[
                                       "merge"],
                                   tool_tip="Whether to merge rois with similar time traces"),
                               IntInput(display_name="ROI Circuity Threshold:",
                                        program_name="roi_circ_threshold",
                                        on_change_function=lambda x,
                                                                    y: data_handler.change_roi_extraction_param(
                                              x, y),
                                        default_val=
                                          data_handler.roi_extraction_params[
                                              "roi_circ_threshold"],
                                        tool_tip="Thresholds the rois based on how circular they are",
                                        min=0, max=100, step=1),
                               IntInput(display_name="Number of connections:",
                                        program_name="connections",
                                        on_change_function=lambda x,
                                                                  y: data_handler.change_eigen_param(
                                            x, y),
                                        default_val=data_handler.eigen_params[
                                            "connections"],
                                        tool_tip="Number of eigen vectors to generate",
                                        min=1, max=10000, step=1),
                               IntInput(display_name="accuracy:",
                                        program_name="accuracy",
                                        on_change_function=lambda x,
                                                                  y: data_handler.change_eigen_param(
                                            x, y),
                                        default_val=data_handler.eigen_params[
                                            "accuracy"],
                                        tool_tip="Number of eigen vectors to generate",
                                        min=1, max=10000, step=1),
                               IntInput(display_name="Number of knn:",
                                        program_name="knn",
                                        on_change_function=lambda x,
                                                                  y: data_handler.change_eigen_param(
                                            x, y),
                                        default_val=data_handler.eigen_params[
                                            "knn"],
                                        tool_tip="Number of eigen vectors to generate",
                                        min=1, max=10000, step=1),
                               IntInput(display_name="Normalize w k :",
                                        program_name="normalize_w_k",
                                        on_change_function=lambda x,
                                                                  y: data_handler.change_eigen_param(
                                            x, y),
                                        default_val=data_handler.eigen_params[
                                            "normalize_w_k"],
                                        tool_tip="Number of eigen vectors to generate",
                                        min=1, max=10000, step=1),
                               IntInput(display_name="max iter:",
                                        program_name="max_iter",
                                        on_change_function=lambda x,
                                                                  y: data_handler.change_roi_extraction_param(
                                            x, y),
                                        default_val=data_handler.roi_extraction_params[
                                            "max_iter"],
                                        tool_tip="Number of eigen vectors to generate",
                                        min=1, max=10000, step=1),

                               IntInput(display_name="Number of time steps:",
                                        program_name="total_num_time_steps",
                                        on_change_function=lambda x,
                                                                  y: data_handler.change_box_param(
                                            x, y),
                                        default_val=data_handler.box_params[
                                            "total_num_time_steps"],
                                        tool_tip=
                                        "Number of time steps to break" +
                                        "the processing into",
                                        min=1, max=10000, step=1),
                               #
                               ]
                              )
# TODO add fill holes/number rois per spatial box to inputs
