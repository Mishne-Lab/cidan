import logging

from qtpy.QtWidgets import *

from cidan.GUI.Data_Interaction.ROIExtractionThread import ROIExtractionThread
from cidan.GUI.ImageView.ROIImageViewModule import ROIImageViewModule
from cidan.GUI.Inputs.OptionInput import OptionInput
from cidan.GUI.ListWidgets.GraphDisplayWidget import GraphDisplayWidget
from cidan.GUI.ListWidgets.ROIListModule import ROIListModule
from cidan.GUI.ListWidgets.TrialListWidget import TrialListWidget
from cidan.GUI.Tabs.Tab import Tab

logger1 = logging.getLogger("cidan.AnalysisTab")


class AnalysisTab(Tab):
    def __init__(self, main_widget):
        self.main_widget = main_widget
        self.cur_plot_type = "neuron"

        self.image_view = ROIImageViewModule(self.main_widget, self, settings_tab=False)

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
                                           val_list=["Line", "Color Mesh"])
        plot_settings_layout.addWidget(self.plot_type_input)
        self.plot_by_input = OptionInput(display_name="Plot By", program_name="",
                                         on_change_function=lambda x,
                                                                   y: self.deselectRoiTime(),
                                         default_index=0, tool_tip="",
                                         display_tool_tip=False,
                                         val_list=["Neuron", "Trial"])
        plot_settings_layout.addWidget(self.plot_by_input)
        self.time_trace_type = OptionInput("Time Trace Type", "",
                                           lambda x, y: self.deselectRoiTime(),
                                           default_index=0,
                                           tool_tip="Select way to calculate time trace",
                                           val_list=list(
                                               self.data_handler.time_trace_possibilities_functions.keys()))
        plot_settings_layout.addWidget(self.time_trace_type)

        time_trace_settings = QWidget()
        time_trace_settings_layout = QVBoxLayout()
        if not len(self.data_handler.trials_loaded) == 1 and not \
        self.data_handler.dataset_params["single_file_mode"] and not \
        self.data_handler.dataset_params["trial_split"]:
            settings_tabs.addTab(time_trace_settings, "Trial Selector")
        else:
            plot_settings_layout.addWidget(time_trace_settings)
            time_trace_settings.hide()

        settings_tabs.addTab(self.image_view.createSettings(), "Display Settings")
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
        export_overlapping_rois = QPushButton(text="Export Overlapping ROIs")
        export_overlapping_rois.hide()
        export_overlapping_rois.clicked.connect(lambda: self.export_overlap_rois())
        if self.main_widget.dev:
            export_overlapping_rois.show()


        # bottom_half.addWidget(self.plot_widget, stretch=2)
        if self.main_widget.data_handler.rois_loaded:
            self.updateTab()
            # self.plot_widget.set_list_items([self.data_handler.get_time_trace(x) for x in range(20)], [x for x in range(20)], None)
        super().__init__("Analysis",
                         column_1=[self.roi_list_module, button_layout,
                                   export_overlapping_rois, settings_tabs],
                         column_2=[self.image_view, self.plot_widget],
                         column_2_display=True, column2_moveable=True)

    def selectAll(self, select):
        self.update_time = False
        for x in self.roi_list_module.roi_item_list:
            x.check_box.setChecked(select)
        self.update_time = True
        self.deselectRoiTime()
    def selectRoiTime(self, num):
        self.deselectRoiTime()

    def deselectRoiTime(self):
        if any(self.data_handler.roi_time_trace_need_update):
            self.main_widget.console.updateText(
                "Some time traces are out of date, please recalculate",
                warning=True)
        if self.update_time:
            if self.plot_by_input.current_state() == "Neuron":
                if self.cur_plot_type != "neuron":
                    self.cur_plot_type = "neuron"
                    self.roi_list_module.select_multiple = True

                try:
                    data_list = []
                    roi_names = []
                    for num2, x in zip(
                            range(1, len(
                                self.roi_list_module.roi_time_check_list) + 1),
                            self.roi_list_module.roi_time_check_list):
                        if x:
                            data_list.append(
                                self.main_widget.data_handler.get_time_trace(num2,
                                                                             trace_type=self.time_trace_type.current_state()))
                            roi_names.append(num2)
                    self.plot_widget.set_list_items(data_list, roi_names, [],
                                                    p_color=self.plot_type_input.current_state() == "Color Mesh",
                                                    type="neuron")
                except AttributeError as e:
                    print("No ROIs have been generated yet")
            else:
                if self.cur_plot_type != "trial":
                    self.roi_list_module.select_multiple = False
                    self.cur_plot_type = "trial"
                    self.selectAll(False)

                try:
                    print(self.roi_list_module.current_selected_roi)
                    if self.roi_list_module.current_selected_roi is not None:
                        roi = self.roi_list_module.current_selected_roi
                        data_list = []
                        roi_names = [roi]
                        for x in self.data_handler.trials_loaded_time_trace_indices:
                            data_list.append(
                                self.main_widget.data_handler.get_time_trace(roi,
                                                                             x))
                        self.plot_widget.set_list_items(data_list, roi_names,
                                                        self.data_handler.trials_loaded_time_trace_indices,
                                                        p_color=self.plot_type_input.current_state() == "Color Mesh",
                                                        type="trial")
                    else:
                        self.plot_widget.set_list_items([], [],
                                                        [],
                                                        p_color=self.plot_type_input.current_state() == "Color Mesh",
                                                        type="trial")
                except AttributeError:
                    print("No ROIs have been generated yet")

    @property
    def data_handler(self):
        return self.main_widget.data_handler

    def update_time_traces(self):
        if (self.main_widget.checkThreadRunning()):
            self.data_handler.calculate_time_traces()
            self.data_handler.update_selected_trials(
                self._time_trace_trial_select_list.selectedTrials())
            self.deselectRoiTime()

    def updateTab(self):
        if (self.main_widget.checkThreadRunning()):
            self._time_trace_trial_select_list.set_items_from_list(
                self.data_handler.trials_all,
                self.data_handler.trials_loaded_time_trace_indices)
            if self.data_handler.rois_loaded:
                self.roi_list_module.set_list_items(self.main_widget.data_handler.rois)
                self.selectAll(False)
                self.image_view.reset_view()
                try:
                    # self.update_time = False
                    for x in self.roi_list_module.roi_item_list[:1]:
                        x.check_box.setChecked(True)
                    # self.deselectRoiTime()
                    # self.update_time = True
                except:
                    pass
            else:
                self.image_view.reset_view()

    def export_overlap_rois(self):
        roi_nums = []
        for num, x in zip(range(1, len(self.roi_list_module.roi_time_check_list) + 1),
                          self.roi_list_module.roi_time_check_list):
            if x:
                roi_nums.append(num)
        if len(roi_nums) == 2:
            self.data_handler.export_overlapping_rois(roi_nums[0], roi_nums[1])
            print("export success")
