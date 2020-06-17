import logging

from qtpy.QtWidgets import *

from CIDAN.GUI.Data_Interaction.ROIExtractionThread import ROIExtractionThread
from CIDAN.GUI.ImageView.ROIImageViewModule import ROIImageViewModule
from CIDAN.GUI.Inputs.OptionInput import OptionInput
from CIDAN.GUI.ListWidgets.GraphDisplayWidget import GraphDisplayWidget
from CIDAN.GUI.ListWidgets.ROIListModule import ROIListModule
from CIDAN.GUI.ListWidgets.TrialListWidget import TrialListWidget
from CIDAN.GUI.Tabs.Tab import Tab

logger1 = logging.getLogger("CIDAN.AnalysisTab")


class AnalysisTab(Tab):
    def __init__(self, main_widget):
        self.main_widget = main_widget
        self.data_handler = main_widget.data_handler
        self.image_view = ROIImageViewModule(self.main_widget, self, True)
        self.roi_list_module = ROIListModule(main_widget.data_handler, self,
                                             select_multiple=True, display_time=False)
        self.thread = ROIExtractionThread(main_widget, QPushButton(),
                                          self.roi_list_module, self)
        self.main_widget.thread_list.append(self.thread)

        self.update_time = True
        settings_tabs = QTabWidget()
        plot_settings_widget = QWidget()
        plot_settings_layout = QVBoxLayout()
        plot_settings_widget.setLayout(plot_settings_layout)
        settings_tabs.addTab(plot_settings_widget, "Settings")

        self.plot_type_input = OptionInput(display_name="Plot Type", program_name="",
                                           on_change_function=lambda x,
                                                                     y: self.deselectRoiTime(),
                                           default_index=0, tool_tip="",
                                           display_tool_tip=False,
                                           val_list=["Color Mesh", "Line"])
        plot_settings_layout.addWidget(self.plot_type_input)
        self.plot_by_input = OptionInput(display_name="Plot By", program_name="",
                                         on_change_function=lambda x,
                                                                   y: self.deselectRoiTime(),
                                         default_index=0, tool_tip="",
                                         display_tool_tip=False,
                                         val_list=["Neuron", "Trial"])

        time_trace_settings = QWidget()
        time_trace_settings_layout = QVBoxLayout()
        settings_tabs.addTab(time_trace_settings, "Trial Selector")
        time_trace_settings_layout.setContentsMargins(0, 0, 0, 0)
        time_trace_settings.setLayout(time_trace_settings_layout)
        self._time_trace_trial_select_list = TrialListWidget(False)
        self._time_trace_trial_select_list.setMinimumHeight(115)
        self._time_trace_trial_select_list.set_items_from_list(
            self.data_handler.trials_all,
            self.data_handler.trials_loaded_time_trace_indices)
        time_trace_settings_layout.addWidget(self._time_trace_trial_select_list,
                                             stretch=5)
        time_trace_update_button = QPushButton("Update Time Traces")

        time_trace_settings_layout.addWidget(time_trace_update_button)
        time_trace_update_button.clicked.connect(
            lambda x: self.update_time_traces())

        self.plot_widget = GraphDisplayWidget(self.main_widget)
        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.clicked.connect(lambda x: self.selectAll(False))
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(lambda x: self.selectAll(True))
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_all_button)
        button_layout.addWidget(self.deselect_all_button)

        # bottom_half.addWidget(self.plot_widget, stretch=2)
        if self.main_widget.data_handler.rois_loaded:
            self.thread.endThread(True)
            # self.plot_widget.set_list_items([self.data_handler.get_time_trace(x) for x in range(20)], [x for x in range(20)], None)
        super().__init__("Analysis",
                         column_1=[self.roi_list_module, button_layout, settings_tabs],
                         column_2=[self.image_view, self.plot_widget],
                         column_2_display=True, column2_moveable=True)

    def selectAll(self, select):
        self.update_time = False
        for x in self.roi_list_module.roi_item_list:
            x.check_box.setChecked(select)
        self.update_time = True
        self.deselectRoiTime()
    def selectRoiTime(self, num):
        if self.update_time:
            try:
                data_list = []
                roi_names = []
                for num2, x in zip(
                        range(1, len(self.roi_list_module.roi_time_check_list)),
                        self.roi_list_module.roi_time_check_list):
                    if x:
                        data_list.append(
                            self.main_widget.data_handler.get_time_trace(num2))
                        roi_names.append(num2)
                self.plot_widget.set_list_items(data_list, roi_names, [],
                                                p_color=self.plot_type_input.current_state() == "Color Mesh",
                                                type="neuron")

            except AttributeError:
                print("No ROIs have been generated yet")

    def deselectRoiTime(self):
        if self.update_time:
            try:
                data_list = []
                roi_names = []
                for num2, x in zip(
                        range(1, len(self.roi_list_module.roi_time_check_list)),
                        self.roi_list_module.roi_time_check_list):
                    if x:
                        data_list.append(
                            self.main_widget.data_handler.get_time_trace(num2))
                        roi_names.append(num2)
                self.plot_widget.set_list_items(data_list, roi_names, [],
                                                p_color=self.plot_type_input.current_state() == "Color Mesh",
                                                type="neuron")
            except AttributeError:
                print("No ROIs have been generated yet")

    def update_time_traces(self):
        if (self.data_handler.time_trace_params[
            "time_trace_type"] != self.time_trace_type.current_state()):
            self.data_handler.time_trace_params[
                "time_trace_type"] = self.time_trace_type.current_state()
            self.data_handler.calculate_time_traces()
        self.data_handler.update_selected_trials(
            self._time_trace_trial_select_list.selectedTrials())
        self.deselectRoiTime()
