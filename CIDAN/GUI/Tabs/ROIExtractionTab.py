import logging

import numpy as np
import pyqtgraph as pg
from qtpy.QtWidgets import *

from CIDAN.GUI.Data_Interaction.ROIExtractionThread import ROIExtractionThread
from CIDAN.GUI.ImageView.ROIPaintImageViewModule import ROIPaintImageViewModule
from CIDAN.GUI.Inputs.IntInput import IntInput
from CIDAN.GUI.Inputs.OptionInput import OptionInput
from CIDAN.GUI.ListWidgets.ROIListModule import ROIListModule
from CIDAN.GUI.ListWidgets.TrialListWidget import TrialListWidget
from CIDAN.GUI.SettingWidget.SettingsModule import roi_extraction_settings
from CIDAN.GUI.Tabs.Tab import Tab
from CIDAN.LSSC.functions.roi_extraction import combine_rois

logger1 = logging.getLogger("CIDAN.ROIExtractionTab")


class ROIExtractionTab(Tab):
    """Class controlling the ROI Extraction tab, inherits from Tab


        Attributes
        ----------
        main_widget : MainWidget
            A reference to the main widget
        data_handler : DataHandler
            A reference to the main DataHandler of MainWidget
        time_plot : pg.PlotWidget
            the plot for the time traces
        roi_list_module : ROIListModule
            The module the controlls the list of ROIs
        thread : ROIExtractionThread
            The thread that runs the roi extraction process

        """

    def __init__(self, main_widget):

        self.main_widget = main_widget
        self.image_view = ROIPaintImageViewModule(main_widget, self, False)
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # This part creates the top left settings/roi list view in two tabs
        self.tab_selector_roi = QTabWidget()
        self.tab_selector_roi.setStyleSheet("QTabWidget {font-size: 20px;}")

        # ROI modification Tab
        roi_modification_tab = QWidget()
        roi_modification_tab.setStyleSheet("margin:0px; padding: 0px;")

        roi_modification_tab_layout = QVBoxLayout()

        roi_modification_tab_layout.setContentsMargins(2, 2, 2, 2)
        roi_modification_tab_layout_top = QHBoxLayout()
        roi_modification_tab_layout_top.setContentsMargins(0, 0, 0, 0)
        roi_modification_tab_layout.addLayout(roi_modification_tab_layout_top)
        display_settings = self.image_view.createSettings()
        display_settings.setStyleSheet("QWidget {border: 2px solid #32414B;}")

        roi_modification_tab_layout.addWidget(display_settings)
        roi_modification_tab.setLayout(roi_modification_tab_layout)

        self.roi_list_module = ROIListModule(main_widget.data_handler, self,
                                             select_multiple=False, display_time=False)
        roi_modification_tab_layout.addWidget(self.roi_list_module)
        roi_modification_button_top_layout = QVBoxLayout()
        roi_modification_button_top_widget = QWidget()
        roi_modification_button_top_widget.setLayout(roi_modification_button_top_layout)
        # roi_modification_tab_layout.addLayout(roi_modification_button_top_layout)

        add_new_roi = QPushButton(text="New ROI from Selection")
        add_new_roi.clicked.connect(lambda x: self.add_new_roi())
        add_to_roi = QPushButton(text="Add to Selected ROI")
        add_to_roi.clicked.connect(
            lambda x: self.modify_roi(self.roi_list_module.current_selected_roi, "add"))

        sub_to_roi = QPushButton(text="Subtract from Selected ROI")
        sub_to_roi.clicked.connect(
            lambda x: self.modify_roi(self.roi_list_module.current_selected_roi,
                                      "subtract"))
        delete_roi = QPushButton(text="Delete Selected ROI")
        delete_roi.clicked.connect(
            lambda x: self.delete_roi(self.roi_list_module.current_selected_roi))

        roi_modification_button_top_layout.addWidget(add_to_roi)
        roi_modification_button_top_layout.addWidget(sub_to_roi)
        roi_modification_button_top_layout.addWidget(add_new_roi)
        roi_modification_button_top_layout.addWidget(delete_roi)
        add_to_roi.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        sub_to_roi.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        add_new_roi.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        delete_roi.setStyleSheet("QWidget {border: 0px solid #32414B;}")

        # Paint Selection button group
        painter_button_group = QButtonGroup()
        off_button = QRadioButton(text="Off")
        off_button.setChecked(True)
        on_button = QRadioButton(text="Add to Selection")
        sub_button = QRadioButton(text="Subtract from Selection")
        magic_wand = QRadioButton(text="Magic Wand")
        off_button.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        on_button.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        sub_button.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        magic_wand.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        painter_button_group.addButton(off_button)
        painter_button_group.addButton(on_button)
        painter_button_group.addButton(sub_button)
        painter_button_group.addButton(magic_wand)
        off_button.clicked.connect(
            lambda x: self.image_view.setSelectorBrushType("off"))
        on_button.clicked.connect(
            lambda x: self.image_view.setSelectorBrushType("add"))
        sub_button.clicked.connect(
            lambda x: self.image_view.setSelectorBrushType("subtract"))
        magic_wand.clicked.connect(
            lambda x: self.image_view.setSelectorBrushType("magic"))
        painter_widget = QWidget()

        painter_layout = QVBoxLayout()
        painter_widget.setLayout(painter_layout)
        label = QLabel(text="Selector Brush: ")
        label.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        painter_layout.addWidget(label)

        painter_layout.addWidget(off_button)
        painter_layout.addWidget(on_button)
        painter_layout.addWidget(sub_button)
        painter_layout.addWidget(magic_wand)

        self._brush_size_options = OptionInput("Brush Size:", "",
                                               lambda x,
                                                      y: self.image_view.setBrushSize(
                                                   y), 1,
                                               "Sets the brush size",
                                               ["1", "3", "5", "7", "9",
                                                "11", "15", "21", "27",
                                                "35"])
        self._brush_size_options.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        painter_layout.addWidget(self._brush_size_options)
        clear_from_selection = QPushButton(text="Clear Selection")
        clear_from_selection.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        clear_from_selection.clicked.connect(
            lambda x: self.image_view.clearPixelSelection())
        painter_layout.addWidget(clear_from_selection)
        roi_modification_button_top_widget.setStyleSheet(
            "QWidget {border: 2px solid #32414B; font-size: 14px}")
        painter_widget.setStyleSheet("QWidget {border: 2px solid #32414B;}")
        roi_modification_tab_layout_top.addWidget(roi_modification_button_top_widget)
        roi_modification_tab_layout_top.addWidget(painter_widget)

        # ROI Settings Tab
        process_button = QPushButton()
        process_button.setText("Apply Settings")
        self.thread = ROIExtractionThread(main_widget, process_button,
                                          self.roi_list_module, self)
        self.main_widget.thread_list.append(self.thread)
        process_button.clicked.connect(lambda: self.thread.runThread())
        self.roi_settings = QWidget()
        self.roi_settings_layout = QVBoxLayout()
        self.roi_settings_layout.setContentsMargins(2, 2, 2, 2)
        self.roi_settings.setLayout(self.roi_settings_layout)
        settings = roi_extraction_settings(main_widget)

        self.roi_settings_layout.addWidget(settings)
        self.roi_settings_layout.addWidget(process_button)

        # adding the tabs to the window
        self.tab_selector_roi.addTab(self.roi_settings, "ROI Creation")
        self.tab_selector_roi.addTab(roi_modification_tab, "ROI Modification")
        self.tab_selector_roi.setMaximumWidth(435)
        self.tab_selector_roi.setMinimumWidth(435)
        # Eigen vector viewer if dev mode is enabled
        if self.main_widget.dev:
            self.eigen_view = QWidget()
            self.eigen_view_layout = QVBoxLayout()
            self.eigen_view_box_input = IntInput("Box Number", "", None, 0, "", 0, 100,
                                                 1, False)
            self.eigen_view_number_input = IntInput("Vector Number", "", None, 0, "", 0,
                                                    100, 1, False)
            self.eigen_view_trial_input = IntInput("Trial Number", "", None, 0, "", 0,
                                                   100, 1, False)
            view_eigen_vector_button = QPushButton("View Eigen Vector")
            view_eigen_vector_button.clicked.connect(lambda x: self.view_eigen_vector())
            self.eigen_view_layout.addWidget(self.eigen_view_box_input)
            self.eigen_view_layout.addWidget(self.eigen_view_number_input)
            self.eigen_view_layout.addWidget(self.eigen_view_trial_input)
            self.eigen_view_layout.addWidget(view_eigen_vector_button)
            self.eigen_view.setLayout(self.eigen_view_layout)
            self.tab_selector_roi.addTab(self.eigen_view, "View Eigen Vectors")

        # If ROIs are loaded, add them to display

        # Tab selector for the time trace window
        tab_selector_time_trace = QTabWidget()
        tab_selector_time_trace.setStyleSheet("QTabWidget {font-size: 20px;}")
        tab_selector_time_trace.setMaximumHeight(220)
        # plot of time traces
        self.time_plot = pg.PlotWidget()
        self.time_plot.getPlotItem().getViewBox().setMouseEnabled(True, False)
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)

        tab_selector_time_trace.addTab(self.time_plot, "Time Trace Plot")
        # Setting window for time traces
        time_trace_settings = QWidget()
        time_trace_settings_layout = QVBoxLayout()
        time_trace_settings_layout.setContentsMargins(0, 0, 0, 0)
        time_trace_settings.setLayout(time_trace_settings_layout)
        self.time_trace_type = OptionInput("Time Trace Type", "",
                                           lambda x, y: x + y,
                                           default_index=0,
                                           tool_tip="Select way to calculate time trace",
                                           val_list=["Mean Florescence Denoised",
                                                     "Mean Florescence",
                                                     "DeltaF Over F Denoised",
                                                     "DeltaF Over F"])
        time_trace_settings_layout.addWidget(self.time_trace_type,
                                             stretch=1)
        # A list widget to select what trials to calculate/display time traces for
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
        tab_selector_time_trace.addTab(time_trace_settings, "Time Trace Settings")
        self.updateTab()

        super().__init__("ROI Extraction",
                         column_1=[self.tab_selector_roi],
                         column_2=[self.image_view, tab_selector_time_trace],
                         column_2_display=True)

    @property
    def data_handler(self):
        return self.main_widget.data_handler

    def add_new_roi(self):
        """
        Adds a new roi using selection(self.image_view.current_selected_pixels_list)
        """
        if (self.main_widget.checkThreadRunning()):
            if len(self.image_view.current_selected_pixels_list) == 0:
                print("Please select some pixels")
                return
            self.data_handler.rois.append(
                np.array(self.image_view.current_selected_pixels_list))
            self.data_handler.gen_roi_display_variables()
            self.data_handler.time_traces.append([])
            for _ in range(len(self.data_handler.trials_all)):
                self.data_handler.time_traces[-1].append(False)
            self.data_handler.calculate_time_trace(len(self.data_handler.rois))
            self.image_view.clearPixelSelection()
            self.update_roi(False)
            self.roi_list_module.set_list_items(self.data_handler.rois)
            self.deselectRoiTime()
            self.roi_list_module.set_current_select(len(self.data_handler.rois))
            self.main_widget.tabs[2].updateTab()

    def delete_roi(self, roi_num):
        """
        Deletes an roi
        Parameters
        ----------
        roi_num
            roi to delete starts at 1

        Returns
        -------

        """
        if (self.main_widget.checkThreadRunning()):
            roi_num = roi_num - 1
            try:
                self.data_handler.rois.pop(roi_num)
                self.data_handler.gen_roi_display_variables()
                self.data_handler.time_traces.pop(roi_num)
                self.update_roi(False)
                self.roi_list_module.set_list_items(self.data_handler.rois)
                self.deselectRoiTime()
                self.main_widget.tabs[2].updateTab()
            except IndexError:
                print("Invalid ROI Selected")

    def modify_roi(self, roi_num, add_subtract="add"):
        """
        Add/subtracts the currently selected pixels from an ROI
        Parameters
        ----------
        roi_num roi to modify starting at 1
        add_subtract either add or subtract depending on operation wanted

        Returns
        -------
        Nothing
        """
        if (self.main_widget.checkThreadRunning()):
            if roi_num is None:
                print("Please select an roi")
                return False
            roi_num = roi_num - 1
            if len(self.image_view.current_selected_pixels_list) == 0:
                print("Please select some pixels")
                return False
            if add_subtract == "add":
                print("Adding Selection to ROI #" + str(roi_num + 1))
                self.data_handler.rois[roi_num] = combine_rois(
                    self.data_handler.rois[roi_num],
                    self.image_view.current_selected_pixels_list)
                self.data_handler.gen_roi_display_variables()
                self.data_handler.calculate_time_trace(roi_num)

            if add_subtract == "subtract":
                print("Subtracting Selection from ROI #" + str(roi_num + 1))
                self.data_handler.rois[roi_num] = [x for x in
                                                   self.data_handler.rois[roi_num]
                                                   if
                                                   x not in self.image_view.current_selected_pixels_list]
                self.data_handler.gen_roi_display_variables()
                self.data_handler.calculate_time_trace(roi_num)
            self.deselectRoiTime()
            self.roi_list_module.set_current_select(roi_num + 1)
            self.image_view.clearPixelSelection()
            self.update_roi(new=False)

            return True

    def update_roi(self, new=True):
        """Resets the roi image display"""
        if (self.main_widget.checkThreadRunning()):
            self.image_view.reset_view(new=new)

    def selectRoiTime(self, num):
        if (self.main_widget.checkThreadRunning()):
            try:
                color_roi = self.main_widget.data_handler.color_list[
                    (num - 1) % len(self.main_widget.data_handler.color_list)]

                if (self.roi_list_module.roi_time_check_list[num - 1]):
                    pen = pg.mkPen(color=color_roi, width=3)
                    self.time_plot.plot(
                        self.main_widget.data_handler.get_time_trace(num),
                        pen=pen)
                    self.time_plot.enableAutoRange(axis=0)
            except AttributeError:
                pass

    def deselectRoiTime(self):
        if (self.main_widget.checkThreadRunning()):
            try:
                self.time_plot.clear()
                self.time_plot.enableAutoRange(axis=0)
                for num2, x in zip(
                        range(1, len(self.roi_list_module.roi_time_check_list)),
                        self.roi_list_module.roi_time_check_list):
                    if x:
                        color_roi = self.main_widget.data_handler.color_list[
                            (num2 - 1) % len(self.main_widget.data_handler.color_list)]

                        pen = pg.mkPen(color=color_roi, width=3)
                        self.time_plot.plot(
                            self.main_widget.data_handler.get_time_trace(num2), pen=pen)
            except AttributeError:
                print("No ROIs have been generated yet")

    def update_time_traces(self):
        if self.main_widget.checkThreadRunning():
            curr_type = "Mean" if "Mean" in self.time_trace_type.current_state() \
                else "DeltaF Over F"
            denoise = "Denoised" in self.time_trace_type.current_state()
            if (self.data_handler.time_trace_params[
                "time_trace_type"] != curr_type or self.data_handler.time_trace_params[
                "denoise"] != denoise):
                self.data_handler.time_trace_params[
                    "time_trace_type"] = curr_type
                self.data_handler.time_trace_params["denoise"] = denoise
                self.data_handler.calculate_time_traces()
            self.data_handler.update_selected_trials(
                self._time_trace_trial_select_list.selectedTrials())
            self.deselectRoiTime()

    def view_eigen_vector(self):
        vector_num = self.eigen_view_number_input.current_state()
        trial_num = self.eigen_view_trial_input.current_state()
        box_num = self.eigen_view_box_input.current_state()
        try:
            vector = self.data_handler.get_eigen_vector(box_num, trial_num=trial_num,
                                                        vector_num=vector_num)
            self.image_view.image_item.image = vector
            self.image_view.image_item.updateImage(autoLevels=True)
        except FileNotFoundError:
            print("Invalid location")

    def updateTab(self
                  ):
        if (self.main_widget.checkThreadRunning()):
            self._time_trace_trial_select_list.set_items_from_list(
                self.data_handler.trials_all,
                self.data_handler.trials_loaded_time_trace_indices)
            if self.data_handler.rois_loaded:
                self.roi_list_module.set_list_items(self.main_widget.data_handler.rois)

            self.image_view.reset_view()
