from qtpy.QtWidgets import *

from cidan.GUI.Inputs.BoolInput import BoolInput
from cidan.GUI.Inputs.FloatInput import FloatInput
from cidan.GUI.Inputs.IntInput import IntInput
from cidan.GUI.Inputs.IntRangeInput import IntRangeInput


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
        # self.setMaximumWidth(300)
        self.setLayout(self.layout)


def filter_setting_block(main_widget):
    data_handler = main_widget.data_handler
    return SettingBlockModule("Filter Settings",
                              [BoolInput(display_name="Local Spatial Denoising:",
                                         program_name="localSpatialDenoising",
                                         on_change_function=lambda x,
                                                                   y: data_handler.change_filter_param(
                                             x, y),
                                         default_val=data_handler.filter_params[
                                             "localSpatialDenoising"],
                                         tool_tip="Local Spatial Denoising helps smooth out the noise in the data by averaging ",
                                         ),

                               BoolInput(
                                   display_name="Z-score:",
                                   program_name="z_score",
                                   on_change_function=lambda x,
                                                             y: data_handler.change_filter_param(
                                       x, y),
                                   default_val=
                                   data_handler.filter_params["z_score"],
                                   tool_tip="Applies a z-score for each pixel across each of the timesteps",
                               ),
                               BoolInput(display_name="Histogram Equalization Method:",
                                         program_name="hist_eq",
                                         on_change_function=lambda x,
                                                                   y: data_handler.change_filter_param(
                                             x, y),
                                         default_val=data_handler.filter_params[
                                             "hist_eq"],
                                         tool_tip="Applies a histogram equalization method(a more advanced z score",
                                         ),
                               BoolInput(display_name="Median filter:",
                                         program_name="median_filter",
                                         on_change_function=lambda x,
                                                                   y: data_handler.change_filter_param(
                                             x, y),
                                         default_val=data_handler.filter_params[
                                             "median_filter"],
                                         tool_tip="Applies a 3D median filter to each timestep. This helps denoise the image. ",
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
                                   tool_tip="The size of the 3D median filter. Recomended size: 3",
                                   min=1, max=50, step=1),
                               BoolInput(display_name="PCA:",
                                         program_name="pca",
                                         on_change_function=lambda x,
                                                                   y: data_handler.change_filter_param(
                                             x, y),
                                         default_val=data_handler.filter_params[
                                             "pca"],
                                         tool_tip="Applies PCA decomposition to the dataset (PCA runs after other filters). \n Both speeds along other calculations and helps remove background noise. \n Time traces are still calculated on filtered stack not PCA stack.",
                                         ),
                               # FloatInput(
                               #     display_name="PCA expression threshold",
                               #     program_name="pca_threshold",
                               #     on_change_function=lambda x,
                               #                               y: data_handler.change_filter_param(
                               #         x, y),
                               #     default_val=
                               #     data_handler.filter_params[
                               #         "pca_threshold"],
                               #     tool_tip="The percentage of the variance that the PCA will express",
                               #     min=0.001, max=.999, step=.001),
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
                                             tool_tip="Crop rows",
                                             min=0, max=10000, step=1),
                               IntRangeInput(display_name="Crop Y:",
                                             # different because the way we display images is weird x is first dim y is second dim
                                             program_name="crop_x",
                                             on_change_function=lambda x,
                                                                       y: data_handler.change_dataset_param(
                                                 x, y),
                                             default_val=data_handler.dataset_params[
                                                 "crop_x"],
                                             tool_tip="Crop columns",
                                             min=0, max=100000, step=1),
                               ] + ([BoolInput(
                                  display_name="Split into Time Blocks(recomended):",
                                  program_name="trial_split",
                                  on_change_function=lambda x,
                                                            y: data_handler.change_dataset_param(
                                      x, y),
                                  default_val=data_handler.dataset_params[
                                      "trial_split"],
                                  tool_tip="Splits the timesteps into separate time blocks better for processing "
                                           "every x timestep. This setting is recomended for all large datasets",
                                  display_tool_tip=False),
                                                                      IntInput(
                                                                          display_name="Time Block Length",
                                                                          program_name="trial_length",
                                                                          on_change_function=lambda x,
                                                                             y: data_handler.change_dataset_param(
                                                       x, y),
                                                                          default_val=
                                                   data_handler.dataset_params[
                                                       "trial_length"],
                                                                          tool_tip="Length of each timeblock",
                                                                          display_tool_tip=False,
                                                                          min=25,
                                                                          max=10000,
                                                                          step=1)] if
                                                                  data_handler.single_dataset_mode else []))


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
            tool_tip="Used to break the calculation into smaller overlapping parts.\n "
                     "This number must be square. The number of spatial boxes per side will be equal to square of this number.",
            min=1, max=10000, step=1)
        ,
        IntInput(
            display_name="Spatial overlap:",
            program_name="spatial_overlap",
            on_change_function=lambda x, y: data_handler.change_box_param(x, y),
            default_val=
            data_handler.box_params[
                "spatial_overlap"],
            tool_tip="Number of pixels to overlap each box.",
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
                                      tool_tip="Max number of ROIs to select for each spatial box, increase this number if not detecting all ROIs",
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
                                      min=0, max=1, step=.01),

                                  #
                                  IntInput(display_name="ROI Eccentricity Threshold:",
                                           program_name="roi_circ_threshold",
                                           on_change_function=lambda x,
                                                                     y: data_handler.change_roi_extraction_param(
                                               x, y),
                                           default_val=
                                           data_handler.roi_extraction_params[
                                               "roi_circ_threshold"],
                                           tool_tip="Thresholds the rois based on how circular they are. Lowering the number would allow more ROIs through",
                                           min=0, max=100, step=1),
                              ]
                              )


def roi_advanced_settings_block(main_widget):
    # TODO fix input into spatial box number
    data_handler = main_widget.data_handler
    input = [IntInput(display_name="Number of eigen vectors:",
                      program_name="num_eig",
                      on_change_function=lambda x,
                                                                  y: data_handler.change_eigen_param(
                                            x, y),
                      default_val=data_handler.eigen_params[
                                            "num_eig"],
                      tool_tip="Number of eigen vectors to generate. Increase if eigen norm image isn't showing all rois",
                      min=1, max=10000, step=1),

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

             IntInput(display_name="Eigen Accuracy:",
                      program_name="eigen_accuracy",
                      on_change_function=lambda x,
                                                y: data_handler.change_eigen_param(
                          x, y),
                      default_val=data_handler.eigen_params[
                          "eigen_accuracy"],
                      tool_tip="10^-x accuracy for each eigen vector, If there are lines in the eigen norm image increase this to 7 or 8",
                      min=1, max=10000, step=1),
             IntInput(display_name="Number of iterations:",
                      program_name="max_iter",
                      on_change_function=lambda x,
                                                y: data_handler.change_roi_extraction_param(
                          x, y),
                      default_val=data_handler.roi_extraction_params[
                          "max_iter"],
                      tool_tip="The number of iterations of the algorithm to preform. Increase if not detecting all rois but they are present in the eigen norm image",
                      min=1, max=10000, step=1),

             ]
    if main_widget.dev:
        input += [BoolInput(
            display_name="Local Max Methods:",
            program_name="local_max_method",
            on_change_function=lambda x,
                                      y: data_handler.change_roi_extraction_param(
                x, y),
            default_val=
            data_handler.roi_extraction_params[
                "local_max_method"],
            tool_tip=""),
            BoolInput(
                display_name="Fill holes:",
                program_name="fill_holes",
                on_change_function=lambda x,
                                          y: data_handler.change_roi_extraction_param(
                    x, y),
                default_val=
                data_handler.roi_extraction_params[
                    "fill_holes"],
                tool_tip="Whether to fill holes in each roi"),

            FloatInput(display_name="Eigen threshold value:",
                       program_name="eigen_threshold_value",
                       on_change_function=lambda x,
                                                 y: data_handler.change_roi_extraction_param(
                           x, y),
                       default_val=
                       data_handler.roi_extraction_params[
                           "eigen_threshold_value"],
                       tool_tip="Number of eigen vectors to select for each ROI. Multiply number of trials by the number of eigen vectors \n if this number is below 300, then it might help to increase this value to .3 or .5",
                       min=0, max=1, step=.01),
            FloatInput(display_name="ROI Eccentricity value:",
                       program_name="roi_eccentricity_limit",
                       on_change_function=lambda x,
                                                 y: data_handler.change_roi_extraction_param(
                           x, y),
                       default_val=
                       data_handler.roi_extraction_params[
                           "roi_eccentricity_limit"],
                       tool_tip="",
                       min=0, max=1, step=.01),
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
                     tool_tip="Number of eigen vectors to generate", min=1, max=10000,
                     step=1),
            BoolInput(
                display_name="Run refinement step:",
                program_name="refinement",
                on_change_function=lambda x,
                                          y: data_handler.change_roi_extraction_param(
                    x, y),
                default_val=
                data_handler.roi_extraction_params[
                    "refinement"],
                tool_tip="Whether to run the refinement step, greatly improves results"),
        ]
    return SettingBlockModule("Advanced Settings", input)
# TODO add fill holes/number rois per spatial box to inputs
