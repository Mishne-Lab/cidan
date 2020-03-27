from GUI.Module import Module
from PySide2.QtWidgets import *
from GUI.Input import *
class SettingBlockModule(Module):
    def __init__(self, name, input_list):
        super().__init__(1)
        self.name = name
        self.input_list = input_list
        self.layout = QVBoxLayout()
        #Type options are int, float, bool, and str
        for input in input_list:
            self.layout.addWidget(input)

        self.setLayout(self.layout)

def filter_setting_block(main_widget):
    data_handler = main_widget.data_handler
    return SettingBlockModule("Filter Settings", [BoolInput(display_name="Median Filter:",
                                                           program_name="median_filter",
                                                           on_change_function=lambda x,y:data_handler.change_filter_param(x,y),
                                                           default_val=data_handler.filter_params["median_filter"],
                                                           tool_tip="Whether to apply a median filter to each timestep",
                                                           ),
                                                  IntInput(
                                                      display_name="Median Filter Size:",
                                                      program_name="median_filter_size",
                                                      on_change_function=lambda x,y:data_handler.change_filter_param(x,y),
                                                      default_val=
                                                      data_handler.filter_params[
                                                          "median_filter_size"],
                                                      tool_tip="The size of the median filter for each timestep",
                                                      min=1,max=50,step=1),
                                                  BoolInput(
                                                      display_name="Z-Score:",
                                                      program_name="z_score",
                                                      on_change_function=lambda x,y:data_handler.change_filter_param(x,y),
                                                      default_val=
                                                      data_handler.filter_params["z_score"],
                                                      tool_tip="Whether to apply a z-score for each pixel across all the timesteps",
                                                      )
                                                  ])

def dataset_setting_block(main_widget):
    data_handler = main_widget.data_handler
    return SettingBlockModule("Dataset Settings",[BoolInput(display_name="Slice Stack:",
                                                   program_name="slice_stack",
                                                    on_change_function=lambda x,y:data_handler.change_dataset_param(x,y),
                                                    default_val=data_handler.dataset_params["slice_stack"],
                                                    tool_tip="Used to Select only "
                                                             "certain timepoints from a dataset"),
                                                  IntInput(display_name="Slice Every:",
                                                           program_name="slice_every",
                                                           on_change_function=lambda x,y:data_handler.change_dataset_param(x,y),
                                                           default_val=data_handler.dataset_params["slice_every"],
                                                           tool_tip="Select every x timesteps",
                                                           min=1, max=100,step=1),
                                                  IntInput(display_name="Slice Start:",
                                                           program_name="slice_start",
                                                           on_change_function=lambda x,y:data_handler.change_dataset_param(x,y),
                                                           tool_tip="Start selecting on x timestep",
                                                           default_val=data_handler.dataset_params["slice_start"],
                                                           min=1, max=1000,step=1)
                                                  ])

def multiprocessing_settings_block(main_widget):
    data_handler = main_widget.data_handler
    return SettingBlockModule("Multiprocessing Settings", [IntInput(display_name="Number of Timesteps:",
                                                           program_name="total_num_time_steps",
                                                           on_change_function=lambda x,y:data_handler.change_box_param(x,y),
                                                           default_val=data_handler.box_params["total_num_time_steps"],
                                                           tool_tip="Number of timesteps to break the processing into",
                                                           min=1, max=1000,step=1),
                                                           IntInput(
                                                               display_name="Number of Spatial Boxes:",
                                                               program_name="total_num_spatial_boxes",
                                                               on_change_function=lambda x,y: data_handler.change_box_param(
                                                                   x, y),
                                                               default_val=
                                                               data_handler.box_params[
                                                                   "total_num_spatial_boxes"],
                                                               tool_tip="Number of boxes to break the calculation into, please make sure it is a sqaure",
                                                               min=1, max=1000, step=1)
                                                           ,
                                                           IntInput(
                                                              display_name="Spatial Overlap:",
                                                              program_name="spatial_overlap",
                                                              on_change_function=lambda x, y: data_handler.change_box_param(x, y),
                                                              default_val=
                                                              data_handler.box_params[
                                                                  "spatial_overlap"],
                                                              tool_tip="Number of pixels to overlap each box",
                                                              min=1, max=1000, step=1)]
                                                )