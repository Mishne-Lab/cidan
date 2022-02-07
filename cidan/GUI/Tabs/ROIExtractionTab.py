import logging
from math import ceil, floor

import numpy as np
import pyqtgraph as pg
import qdarkstyle
from qtpy.QtWidgets import *

from cidan.GUI.Data_Interaction import DataHandler
from cidan.GUI.Data_Interaction.ROIExtractionThread import ROIExtractionThread
from cidan.GUI.Data_Interaction.TimeTraceCalculateThread import TimeTraceCalculateThread
from cidan.GUI.ImageView.ROIPaintImageViewModule import ROIPaintImageViewModule
from cidan.GUI.Inputs.IntInput import IntInput
from cidan.GUI.Inputs.OptionInput import OptionInput
from cidan.GUI.ListWidgets.ROIListModule import ROIListModule
from cidan.GUI.ListWidgets.TrialListWidget import TrialListWidget
from cidan.GUI.SettingWidget.SettingsModule import roi_extraction_settings
from cidan.GUI.Tabs.Tab import Tab
from cidan.LSSC.functions.roi_extraction import combine_rois, \
    number_connected_components, fill_holes_func

logger1 = logging.getLogger("cidan.ROIExtractionTab")


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
        self.roi_unconnected = QErrorMessage(self.main_widget.main_window)
        self.merge_mode = False  # this is for the multi merge capability

        # This part creates the top left settings/roi list view in two tabs
        self.tab_selector_roi = QTabWidget()
        self.tab_selector_roi.keyPressEvent = self.keyPressEvent
        # self.tab_selector_roi.setStyleSheet("QTabWidget {font-size: 20px;}")

        # ROI modification Tab
        roi_modification_tab = QWidget()
        roi_modification_tab.setStyleSheet("margin:0px; padding: 0px;")

        roi_modification_tab_layout = QVBoxLayout()

        roi_modification_tab_layout.setContentsMargins(1, 1, 1, 1)
        roi_modification_tab_layout_top = QVBoxLayout()
        roi_modification_tab_layout_top.setContentsMargins(0, 0, 0, 0)
        roi_modification_tab_layout.addLayout(roi_modification_tab_layout_top)
        display_settings = self.image_view.createSettings()
        display_settings.setStyleSheet("QWidget {border: 2px solid #32414B;}")

        roi_modification_tab_layout.addWidget(display_settings)
        roi_modification_tab.setLayout(roi_modification_tab_layout)

        self.roi_list_module = ROIListModule(main_widget.data_handler, self,
                                             select_multiple=False, display_time=False)
        roi_modification_tab_layout.addWidget(self.roi_list_module)
        roi_modification_button_top_layout = QHBoxLayout()
        roi_modification_button_top_layout.setContentsMargins(2, 2, 2, 2)
        roi_modification_button_top_widget = QWidget()
        roi_modification_button_top_widget.setStyleSheet(
            "QWidget {border: 0px solid #32414B;}")
        roi_modification_button_top_widget.setLayout(roi_modification_button_top_layout)
        # roi_modification_tab_layout.addLayout(roi_modification_button_top_layout)

        add_new_roi = QPushButton(text="Create ROI from\nMask (D)")
        add_new_roi.clicked.connect(lambda x: self.add_new_roi())
        add_new_roi.setToolTip("Use this button to create a new ROI from mask. \n"
                               "ROI is added to bottiom of ROI list")
        add_to_roi = QPushButton(text="Add Mask to \nSelected ROI (A)")
        add_to_roi.clicked.connect(
            lambda x: self.modify_roi(self.roi_list_module.current_selected_roi, "add"))
        add_to_roi.setToolTip("Use this button to add the current mask to"
                              " the selected ROI. \n"
                              "Select an roi in the ROI list below")
        sub_to_roi = QPushButton(text="Subtract Mask from\nSelected ROI (S)")
        sub_to_roi.clicked.connect(
            lambda x: self.modify_roi(self.roi_list_module.current_selected_roi,
                                      "subtract"))
        sub_to_roi.setToolTip("Use this button to subtract the current mask from"
                              " the selected ROI. \n"
                              "Select an roi in the ROI list below")
        delete_roi = QPushButton(text="Delete Selected\nROI (F)")
        delete_roi.setToolTip("Use this button to delete the selected ROI. \n"
                              "Select an roi in the ROI list below")
        delete_roi.clicked.connect(
            lambda x: self.delete_roi(self.roi_list_module.current_selected_roi))
        self.merge_button = QPushButton(text="Merge Selected")
        self.merge_button.setToolTip("this merges the currently selected rois together")
        self.merge_button.clicked.connect(
            lambda x: self.merge_rois())
        self.merge_button.hide()
        roi_modification_button_top_layout.addWidget(add_new_roi)
        roi_modification_button_top_layout.addWidget(add_to_roi)
        roi_modification_button_top_layout.addWidget(sub_to_roi)

        roi_modification_button_top_layout.addWidget(delete_roi)
        roi_modification_button_top_layout.addWidget(self.merge_button)
        add_to_roi.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        sub_to_roi.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        add_new_roi.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        delete_roi.setStyleSheet("QWidget {border: 0px solid #32414B;}")

        # Paint Selection button group
        painter_button_group = QButtonGroup()
        self.off_button = QRadioButton(text="Off (Q)")
        self.off_button.setChecked(True)
        self.off_button.setToolTip("Turns off the mask brush")
        self.on_button = QRadioButton(text="Add to Mask (W)")
        self.on_button.setToolTip(
            "Turns on the selector brush, draw on the image by right clicking")
        self.sub_button = QRadioButton(text="Subtract from Mask (E)")
        self.sub_button.setToolTip(
            "Turns the selector brush to subtract mode. Removing currently selected pixels")
        self.magic_wand = QRadioButton(text="Magic Wand (R)")
        self.magic_wand.setToolTip(
            "Turns the Mask brush to magic wand mode. Click on where you believe \n"
            "there should be an ROI, and it will attempt to create one for you.\n Note just click one at a time. ")
        self.off_button.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        self.on_button.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        self.sub_button.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        self.magic_wand.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        painter_button_group.addButton(self.off_button)
        painter_button_group.addButton(self.on_button)
        painter_button_group.addButton(self.sub_button)
        painter_button_group.addButton(self.magic_wand)
        self.off_button.clicked.connect(
            lambda x: self.image_view.setSelectorBrushType("off"))
        self.on_button.clicked.connect(
            lambda x: self.image_view.setSelectorBrushType("add"))
        self.sub_button.clicked.connect(
            lambda x: self.image_view.setSelectorBrushType("subtract"))
        self.magic_wand.clicked.connect(
            lambda x: self.image_view.setSelectorBrushType("magic"))
        painter_widget = QWidget()

        painter_layout = QVBoxLayout()
        painter_widget.setLayout(painter_layout)
        label = QLabel(text="Mask Brush (green): ")
        label.setStyleSheet("QWidget {border: 0px solid #32414B;}")

        painter_layout_sub_1 = QHBoxLayout()
        painter_layout_sub_1.addWidget(label)
        painter_layout_sub_1.addWidget(self.off_button)
        painter_layout_sub_2 = QHBoxLayout()
        painter_layout_sub_2.addWidget(self.on_button)
        painter_layout_sub_2.addWidget(self.magic_wand)
        painter_layout_sub_3 = QHBoxLayout()
        painter_layout_sub_3.addWidget(self.sub_button)

        painter_layout.addLayout(painter_layout_sub_1)
        painter_layout.addLayout(painter_layout_sub_2)
        painter_layout.addLayout(painter_layout_sub_3)

        self._brush_size_options = OptionInput("Brush Size:", "",
                                               lambda x,
                                                      y: self.image_view.setBrushSize(
                                                   y), 1,
                                               "Sets the brush size",
                                               ["1", "3", "5", "7", "9",
                                                "11", "15", "21", "27",
                                                "35"])
        self._brush_size_options.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        painter_layout_sub_3.addWidget(self._brush_size_options)
        clear_from_selection = QPushButton(text="Clear Mask (T)")
        clear_from_selection.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        clear_from_selection.clicked.connect(
            lambda x: self.image_view.clearPixelSelection(display_update_text=True))
        painter_layout.addWidget(clear_from_selection)
        self.select_all_screen_button = QPushButton(
            text="Select all rois on screen (shift+click)")
        self.select_all_screen_button.setStyleSheet(
            "QWidget {border: 0px solid #32414B;}")
        self.select_all_screen_button.clicked.connect(
            lambda x: self.select_many_rois_box())
        painter_layout.addWidget(self.select_all_screen_button)
        self.select_all_screen_button.hide()
        roi_modification_button_top_widget.setStyleSheet(
            "QWidget {border: 2px solid #32414B; font-size: %dpx}" % (
                    self.main_widget.scale * 20))
        painter_widget.setStyleSheet("QWidget {border: 2px solid #32414B;}")
        roi_modification_tab_layout_top.addWidget(painter_widget)
        roi_modification_tab_layout_top.addWidget(roi_modification_button_top_widget)

        recalc_time_traces_button = QPushButton(text="Recalculate Time Traces")
        recalc_time_traces_button.clicked.connect(lambda: self.update_time_traces())
        self.deselect_all_button = QPushButton("Deselect All")
        self.deselect_all_button.clicked.connect(lambda x: self.selectAll(False))
        self.deselect_all_button.hide()
        recalc_horiz_layout = QHBoxLayout()
        recalc_horiz_layout.addWidget(recalc_time_traces_button)
        recalc_horiz_layout.addWidget(self.deselect_all_button)
        roi_modification_tab_layout.addLayout(recalc_horiz_layout)
        # ROI Settings Tab
        process_button = QPushButton()
        process_button.setText("Apply Settings")
        self.thread = ROIExtractionThread(main_widget, process_button,
                                          self.roi_list_module, self)
        self.main_widget.thread_list.append(self.thread)
        self.time_trace_thread = TimeTraceCalculateThread(main_widget, process_button,
                                                          self.roi_list_module)
        self.main_widget.thread_list.append(self.time_trace_thread)
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
        # self.tab_selector_roi.setMaximumWidth(435)
        # self.tab_selector_roi.setMinimumWidth(435)
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
        # tab_selector_time_trace.setStyleSheet("QTabWidget {font-size: 20px;}")
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
                                           lambda x, y: self.deselectRoiTime(),
                                           default_index=0,
                                           tool_tip="Select way to calculate time trace,"
                                                    " \ncheck github for more details",
                                           val_list=list(
                                               self.data_handler.time_trace_possibilities_functions.keys()))
        time_trace_settings_layout.addWidget(self.time_trace_type,
                                             stretch=1)
        # A list widget to select what trials to calculate/display time traces for
        self._time_trace_trial_select_list = TrialListWidget(False)
        self._time_trace_trial_select_list.setMinimumHeight(115)
        self._time_trace_trial_select_list.set_items_from_list(
            self.data_handler.trials_all,
            self.data_handler.trials_loaded_time_trace_indices)
        if not len(self.data_handler.trials_loaded) == 1 and not \
        self.data_handler.dataset_params["single_file_mode"] and not \
        self.data_handler.dataset_params["trial_split"]:
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
                         column_2_display=True, horiz_moveable=True)
        self._brush_size_options.setMinimumHeight(
            int(30 * ((self.logicalDpiX() / 96.0 - 1) / 2 + 1)))
        self.installEventFilter(self.main_widget.eventFilterCustom)

        for attr in dir(self):
            if isinstance(self.__getattribute__(attr), QWidget):
                self.__getattribute__(attr).installEventFilter(
                    self.main_widget.eventFilterCustom)

    def keyPressAction(self, event):
        if self.tab_selector_roi.currentIndex() == 1:
            if event.key() == 81:  # Q
                self.off_button.setChecked(True)
                self.image_view.setSelectorBrushType("off")
                event.accept()
            if event.key() == 87:  # W
                if self.on_button.isChecked():
                    self.off_button.setChecked(True)
                    self.image_view.setSelectorBrushType("off")
                else:
                    self.on_button.setChecked(True)
                    self.image_view.setSelectorBrushType("add")
                    event.accept()
            if event.key() == 69:  # E
                if self.sub_button.isChecked():
                    self.off_button.setChecked(True)
                    self.image_view.setSelectorBrushType("off")
                else:
                    self.sub_button.setChecked(True)
                    self.image_view.setSelectorBrushType("subtract")
                event.accept()
            if event.key() == 82:  # R
                if self.magic_wand.isChecked():
                    self.off_button.setChecked(True)
                    self.image_view.setSelectorBrushType("off")
                else:
                    self.magic_wand.setChecked(True)
                    self.image_view.setSelectorBrushType("magic")
                event.accept()
            if event.key() == 84:  # T
                self.image_view.clearPixelSelection()
                event.accept()
            if event.key() == 65:  # A
                self.modify_roi(self.roi_list_module.current_selected_roi, "add")
                event.accept()
            if event.key() == 83:  # S
                self.modify_roi(self.roi_list_module.current_selected_roi, "subtract")
                event.accept()
            if event.key() == 68:  # D
                self.add_new_roi()
                event.accept()
            if event.key() == 70 or 16777219 == event.key():  # F or delete
                self.delete_roi(self.roi_list_module.current_selected_roi)
                event.accept()
            if event.key() == 16777234:
                self.roi_list_module.select_roi_next(False)
                event.accept()
            if event.key() == 16777236:
                self.roi_list_module.select_roi_next(True)
                event.accept()

        return event.isAccepted()

    @property
    def data_handler(self):
        return self.main_widget.data_handler

    def add_new_roi(self):
        """
        Adds a new roi using selection(self.image_view.current_selected_pixels_list)
        """
        self.main_widget.console.updateText("Creating new ROI")
        if (self.main_widget.checkThreadRunning()):
            if len(self.image_view.current_selected_pixels_list) == 0:
                print("Please select some pixels")
                self.main_widget.console.updateText("Please select some pixels")
                return
            temp_roi = np.array(self.image_view.current_selected_pixels_list)
            num = number_connected_components(
                self.data_handler.shape[0] * self.data_handler.shape[1],
                [0] + self.data_handler.shape, temp_roi)
            holes_filled = fill_holes_func([temp_roi], self.data_handler.shape[0] *
                                           self.data_handler.shape[1],
                                           [0] + self.data_handler.shape)
            if len(holes_filled[0]) != len(temp_roi):
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setStyleSheet(qdarkstyle.load_stylesheet())

                msg.setText(
                    "Would you like to fill in your selection?")
                # msg.setInformativeText("This is additional information")
                # msg.setWindowTitle("MessageBox demo")
                # msg.setDetailedText("The details are as follows:")
                msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                retval = msg.exec_()
                if retval == 16384:
                    temp_roi = holes_filled[0]
                else:
                    return False
            if num == 2:

                pass
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setStyleSheet(qdarkstyle.load_stylesheet())

                msg.setText(
                    "The pixels you selected are not connected, Are you sure you want to create a new ROI with them?")
                # msg.setInformativeText("This is additional information")
                # msg.setWindowTitle("MessageBox demo")
                # msg.setDetailedText("The details are as follows:")
                msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                retval = msg.exec_()
                if retval == 1024:
                    pass
                else:
                    return False
            # self.data_handler.time_traces.append([])
            # for _ in range(len(self.data_handler.trials_all)):
            #     self.data_handler.time_traces[-1].append(False)
            # self.data_handler.time_traces.append([np.zeros(50)])
            roi_key = self.data_handler.add_new_roi(temp_roi)
            self.image_view.clearPixelSelection()
            self.update_roi(False)
            self.roi_list_module.set_list_items(self.data_handler.rois_dict)
            self.deselectRoiTime()
            self.roi_list_module.set_current_select(roi_key)
            self.main_widget.tabs[2].updateTab()
            self.off_button.setChecked(True)
            self.image_view.setSelectorBrushType("off")
            self.main_widget.console.updateText(
                "Some time traces are out of date, please recalculate",
                warning=True)

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

            if roi_num is None or roi_num <1:
                self.main_widget.console.updateText("Invalid ROI Selected")
                print("Invalid ROI Selected")
                return
            try:
                self.main_widget.console.updateText(
                    "Deleting ROI #%s" % str(roi_num + 1))
                self.data_handler.delete_roi(roi_num, input_key=False)
                self.update_roi(False)
                self.roi_list_module.set_list_items(self.data_handler.rois_dict)
                self.deselectRoiTime()
                self.main_widget.tabs[2].updateTab()
            except IndexError:
                self.main_widget.console.updateText("Invalid ROI Selected")
                print("Invalid ROI Selected")

    def modify_roi(self, roi_num, add_subtract="add", override=False):
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
            self.select_many_rois_box_dev()
            if roi_num is None or roi_num <1:
                print("Please select an roi")
                self.main_widget.console.updateText("Please Select an ROI")
                return False
            if len(self.image_view.current_selected_pixels_list) == 0:
                print("Please select some pixels")
                self.main_widget.console.updateText("Please select some pixels")
                return False
            if add_subtract == "add":
                print("Adding Selection to ROI #" + str(roi_num + 1))
                temp_roi = combine_rois(
                    self.data_handler.rois[roi_num],
                    self.image_view.current_selected_pixels_list)
                num = number_connected_components(
                    self.data_handler.shape[0] * self.data_handler.shape[1],
                    [0] + self.data_handler.shape, temp_roi)
                if num == 2 or override:

                    self.data_handler.update_roi(roi_num, temp_roi, input_key=False)
                    # self.data_handler.gen_roi_display_variables()
                    self.data_handler.roi_time_trace_need_update[roi_num] = True

                else:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setStyleSheet(qdarkstyle.load_stylesheet())

                    msg.setText(
                        "The pixels you selected are not connect to ROI #%s, Are you sure you want to add them?" % str(
                            roi_num + 1))
                    # msg.setInformativeText("This is additional information")
                    # msg.setWindowTitle("MessageBox demo")
                    # msg.setDetailedText("The details are as follows:")
                    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                    retval = msg.exec_()
                    if retval == 1024:
                        self.data_handler.update_roi(roi_num, temp_roi, input_key=False)
                        self.data_handler.roi_time_trace_need_update[roi_num] = True
                    else:
                        return False


            if add_subtract == "subtract":
                print("Subtracting Selection from ROI #" + str(roi_num + 1))
                temp_roi = [x for x in
                            self.data_handler.rois[roi_num]
                            if
                            x not in self.image_view.current_selected_pixels_list]

                self.data_handler.update_roi(roi_num, temp_roi, input_key=False)
                self.data_handler.roi_time_trace_need_update[roi_num] = True
            self.deselectRoiTime()
            self.roi_list_module.set_current_select(roi_num + 1)
            self.image_view.clearPixelSelection()

            self.image_view.setSelectorBrushType("off")
            self.update_roi(new=False)
            self.main_widget.tabs[2].updateTab()
            if add_subtract == "subtract":
                self.main_widget.console.updateText(
                    "Subtracting Selection from ROI #" + str(roi_num + 1))

            if add_subtract == "add":
                self.main_widget.console.updateText(
                    "Adding Selection to ROI #" + str(roi_num + 1))
            self.off_button.setChecked(True)
            self.main_widget.console.updateText(
                "Some time traces are out of date, please recalculate",
                warning=True)
            return True

    def update_roi(self, new=True):
        """Resets the roi image display"""
        if (self.main_widget.checkThreadRunning()):
            self.image_view.reset_view(new=new)

    def selectRoiTime(self, num):

        try:
            color_roi = self.main_widget.data_handler.color_list[
                (num) % len(self.main_widget.data_handler.color_list)]

            if (self.roi_list_module.roi_time_check_list[num]):
                if self.data_handler.roi_time_trace_need_update[num]:
                    self.main_widget.console.updateText(
                        "Some time traces are out of date, please recalculate",
                        warning=True)
                pen = pg.mkPen(color=color_roi, width=3)
                time_trace = self.main_widget.data_handler.get_time_trace(num,
                                                                          trace_type=self.time_trace_type.current_state())
                if type(time_trace) == bool:
                    return
                self.time_plot.plot(time_trace,
                                    pen=pen)
                self.time_plot.enableAutoRange(axis=0)
        except AttributeError:
            pass

    def toggle_merge_mode(self, cur_val):
        self.merge_mode = cur_val
        self.roi_list_module.select_multiple = cur_val
        self.image_view.select_multiple = cur_val
        if self.merge_mode:
            self.merge_button.show()
            self.select_all_screen_button.show()
            self.deselect_all_button.show()
        else:
            self.merge_button.hide()
            self.select_all_screen_button.hide()
            self.deselect_all_button.hide()
    def selectAll(self, select):
        self.update_time = False
        for x in self.roi_list_module.roi_item_list:
            x.check_box.setChecked(select)
        self.update_time = True
        self.deselectRoiTime()
    def merge_rois(self):
        rois_to_merge = [x.id for x in self.roi_list_module.roi_item_list if
                         x.check_box.checkState()]
        index = self.data_handler.merge_rois(rois_to_merge)


        self.image_view.clearPixelSelection()

        self.image_view.setSelectorBrushType("off")
        self.update_roi(new=False)
        self.updateTab()
        self.main_widget.tabs[2].updateTab()
        self.deselectRoiTime()
        self.roi_list_module.set_current_select(index)

    def select_many_rois_box(self, event=None):
        if not self.main_widget.dev or not self.merge_mode:
            return
        if event is not None:
            event.accept()
        region_range = self.image_view.image_view.getView().vb.state["viewRange"]
        image_size = self.main_widget.data_handler.shape
        for y in range(2):
            for h in range(2):
                if h == 1:
                    region_range[y][h] = ceil(region_range[y][h])
                else:
                    region_range[y][h] = floor(region_range[y][h])
        if region_range[0][0] < 0:
            region_range[0][0] = 0
        if region_range[1][0] < 0:
            region_range[1][0] = 0
        if region_range[0][1] > image_size[1]:
            region_range[0][1] = image_size[1]
        if region_range[1][1] > image_size[0]:
            region_range[1][1] = image_size[0]
        region_range = region_range[::-1]
        roi_num_image = np.reshape(self.main_widget.data_handler.pixel_with_rois_flat,
                                   image_size)
        rois_in_region = list(np.unique(
            roi_num_image[region_range[0][0]:region_range[0][1],
            region_range[1][0]:region_range[1][1]]))
        rois_in_region.remove(0)
        for x in rois_in_region:
            self.roi_list_module.set_current_select(int(x), force_on=True)
        # self.updateTab()

        # take range down to image size
        # use that to segment out portion of roiimage view then run np unique on it to find which rois are present
        # then select all those rois, idk how to do this by passing a one time parameter that allows multiple rois to be selected that resets everytime all selections are cleared
        # see how this looks and then add a merge mode to it if its too much and makes user experience worse
        # oh or add a menu item that turns merge mode on and off to the top menu thing

    def deselectRoiTime(self):

        try:
            self.time_plot.clear()
            self.time_plot.enableAutoRange(axis=0)
            for num2, x in enumerate(self.roi_list_module.roi_time_check_list):
                if x:
                    if self.data_handler.roi_time_trace_need_update[num2]:
                        self.main_widget.console.updateText(
                            "Some time traces are out of date, please recalculate",
                            warning=True)
                    color_roi = self.main_widget.data_handler.color_list[
                        (num2) % len(self.main_widget.data_handler.color_list)]

                    pen = pg.mkPen(color=color_roi, width=3)
                    try:
                        self.time_plot.plot(
                            self.main_widget.data_handler.get_time_trace(num2,
                                                                     trace_type=self.time_trace_type.current_state()),
                            pen=pen)
                    except:
                        print("ROI time trace not calculated")
        except AttributeError:
            print("No ROIs have been generated yet")

    def update_time_traces(self):
        def end_func():
            if self.data_handler.real_trials:
                self.data_handler.update_selected_trials(
                    self._time_trace_trial_select_list.selectedTrials())
            self.deselectRoiTime()
            self.main_widget.tabs[2].updateTab()
        if self.main_widget.checkThreadRunning():

            if any(
                self.data_handler.roi_time_trace_need_update):

                # self.data_handler.calculate_time_traces()
                self.time_trace_thread.runThread(end_func)

            else:
                self.main_widget.console.updateText("No update necessary")

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
                self.roi_list_module.set_list_items(
                    self.main_widget.data_handler.rois_dict)

            self.image_view.reset_view()
