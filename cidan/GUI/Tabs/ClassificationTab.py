import logging
import random

import numpy as np
import pyqtgraph as pg
import qdarkstyle
from qtpy.QtWidgets import *
from skimage import measure

from cidan.GUI.Data_Interaction import DataHandler
from cidan.GUI.Data_Interaction.ROIExtractionThread import ROIExtractionThread
from cidan.GUI.ImageView.ROIImageViewModule import ROIImageViewModule
from cidan.GUI.Inputs.ColorInput import ColorInput
from cidan.GUI.Inputs.OptionInput import OptionInput
from cidan.GUI.Inputs.StringInput import StringInput
from cidan.GUI.ListWidgets.ClassListModule import ClassListModule
from cidan.GUI.ListWidgets.ROIListModule import ROIListModule
from cidan.GUI.ListWidgets.TrialListWidget import TrialListWidget
from cidan.GUI.Tabs.Tab import Tab
from cidan.TimeTrace.signal_to_noise import trial_PSNR

logger1 = logging.getLogger("cidan.ROIExtractionTab")


class ClassificationTab(Tab):
    """Class controlling the Classification tab, inherits from Tab


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

        self.image_view = ROIImageViewModule(main_widget, self, False,
                                             display_class_option=True)
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.roi_list_module = ROIListModule(main_widget.data_handler, self,
                                             select_multiple=False, display_time=False,
                                             class_version=True)
        self.image_view.outlines = False
        self.image_view.current_forground = "Class"
        # self.tab_selector_roi.setStyleSheet("QTabWidget {font-size: 20px;}")

        display_setting_layout = QVBoxLayout()
        display_setting_layout.addWidget(QLabel("Display Settings"))
        self.display_settings = self.image_view.createSettings()
        display_setting_layout.addWidget(self.display_settings)
        roi_classification_layout = QVBoxLayout()

        self.button_layout_1 = QHBoxLayout()
        self.button_layout_2 = QHBoxLayout()

        self.stats_layout = QVBoxLayout()
        self.roi_label = QLabel(
            "ROI #{}:".format(self.roi_list_module.current_selected_roi))
        roi_classification_layout.addWidget(self.roi_label)
        stats_layout_h1 = QHBoxLayout()
        stats_layout_h2 = QHBoxLayout()
        self.stat_labels = {"Area": QLabel("Area: None"),
                            "Eccentricity": QLabel("Eccentricity: None"),
                            "Extent": QLabel("Extent: None"),
                            "PSNR": QLabel("PSNR: None")}
        stats_layout_h1.addWidget(self.stat_labels["Area"])
        stats_layout_h1.addWidget(self.stat_labels["Eccentricity"])

        stats_layout_h2.addWidget(self.stat_labels["Extent"])
        stats_layout_h2.addWidget(self.stat_labels["PSNR"])
        # stats_layout_h1.addWidget(QLabel(""))
        # stats_layout_h2.addWidget(QLabel(""))
        self.stats_layout.addLayout(stats_layout_h1)
        self.stats_layout.addLayout(stats_layout_h2)
        roi_classification_layout.addLayout(self.stats_layout)
        roi_classification_layout.addLayout(self.button_layout_1)
        roi_classification_layout.addLayout(self.button_layout_2)

        self.roi_class_dict_tabs = QTabWidget()
        self.roi_class_dict_tabs.addTab(self.roi_list_module, "ROI List")

        self.class_list_module = ClassListModule(classifier_tab=self,
                                                 data_handler=self.data_handler, )
        self.create_class_button = QPushButton("Create New Class")
        self.create_class_button.clicked.connect(lambda x: self.create_class())
        class_layout = QVBoxLayout()
        class_layout.addWidget(self.class_list_module)
        class_layout.addWidget(self.create_class_button)
        class_layout_widget = QWidget()
        class_layout_widget.setLayout(class_layout)

        self.roi_class_dict_tabs.addTab(class_layout_widget, "Classes")

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
        self.roi_list_module.keyPressEvent = self.keyPressEvent
        self.roi_class_dict_tabs.keyPressEvent = self.keyPressEvent
        self.updateTab()

        super().__init__("Classification",
                         column_1=[self.display_settings, roi_classification_layout,
                                   self.roi_class_dict_tabs],
                         column_2=[self.image_view, tab_selector_time_trace],
                         column_2_display=True, horiz_moveable=True)

    def keyPressEvent(self, event):
        super(ClassificationTab, self).keyPressEvent(event)
        print(event.key())
        # add_thing = 49 # key number for key "1"
        # for x in range(0,9,1):
        #     if event.key() == add_thing+x and len(self.data_handler.classes.keys())>x:
        #
        #         self.assign_roi_to_class(list(self.data_handler.classes.keys())[x])
        # if event.key()== add_thing-1 and len(self.data_handler.classes.keys())>=10:# for key 0 which will assign class 10
        #     self.assign_roi_to_class(list(self.data_handler.classes.keys())[9])

    @property
    def data_handler(self):
        return self.main_widget.data_handler

    def set_class_buttons(self):
        def button_function(class_id):
            return lambda x: self.assign_roi_to_class(class_id)

        for i in reversed(range(self.button_layout_1.count())):
            self.button_layout_1.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.button_layout_2.count())):
            self.button_layout_2.itemAt(i).widget().setParent(None)
        ids = list(self.data_handler.classes.keys())
        for num, id in enumerate(ids):
            button = QPushButton(self.data_handler.classes[id]["name"])

            button.clicked.connect(button_function(id))
            if num < 4:
                self.button_layout_1.addWidget(button)
            else:
                self.button_layout_2.addWidget(button)

        self.class_list_module.set_list_items(self.data_handler.classes)
        # button = QPushButton("Delete ROI")
        # button.clicked.connect(lambda x: self.delete_roi())

    def assign_roi_to_class(self, class_name):
        if self.roi_list_module.current_selected_roi is not None:
            try:
                self.data_handler.assign_roi_class(
                    self.roi_list_module.current_selected_roi, new_class=class_name,
                    input_key=False)
            except AttributeError:
                pass
            self.updateTab()
            try:
                next_roi = self.data_handler.classes["Unassigned"]["rois"][0]
                self.image_view.zoomRoi(self.data_handler.rois_dict[next_roi]["index"])
                self.roi_list_module.set_current_select(next_roi)
            except IndexError:
                pass

    def create_class(self):
        if len(self.data_handler.classes.keys()) >= 8:
            msg = QMessageBox()
            msg.setStyleSheet(qdarkstyle.load_stylesheet())
            msg.setWindowTitle("Notice")

            msg.setText(
                "The maximum number of allowed classes is 8")
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()
            return
        id = len(self.data_handler.classes.keys())

        while id in self.data_handler.classes.keys():
            id = random.randint(a=1000, b=100000)
        self.data_handler.classes[str(id)] = {"name": "default name", "rois": [],
                                              "color": self.data_handler.color_list[
                                                  id % len(
                                                      self.data_handler.color_list)],
                                              "editable": True}
        self.edit_class(str(id))

    def delete_class(self, class_id):

        if not self.data_handler.classes[class_id]["editable"]:
            msg = QMessageBox()
            msg.setStyleSheet(qdarkstyle.load_stylesheet())
            msg.setWindowTitle("Warning")

            msg.setText("You can't edit this class")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
            return
        del self.data_handler.classes[class_id]
        self.updateTab()

    def edit_class(self, class_id):
        class_name = self.data_handler.classes[class_id]["name"]
        if not self.data_handler.classes[class_id]["editable"]:
            msg = QMessageBox()
            msg.setStyleSheet(qdarkstyle.load_stylesheet())
            msg.setWindowTitle("Warning")

            msg.setText("You can't edit this class")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()
            return
        dialog = QDialog()
        dialog.layout = QVBoxLayout()
        self.name_input = StringInput(default_val=class_name,
                                      display_name="Class Name:",
                                      on_change_function=None,
                                      program_name="Class Name:",
                                      tool_tip="The name of the class")
        dialog.layout.addWidget(self.name_input)
        self.color_input = ColorInput(
            default_val=self.data_handler.classes[class_id]["color"],
            display_name="Color:", program_name="Color", on_change_function=None,
            tool_tip="")
        dialog.layout.addWidget(self.color_input)
        style = str("""
                    QVBoxLayout {
                
                        border:%dpx;
                    }


                    """ % (2))
        dialog.setStyleSheet(qdarkstyle.load_stylesheet() + style)

        def finish_func(delete=False):
            dialog.close()
            self.data_handler.classes[class_id][
                "color"] = self.color_input.current_state()
            self.data_handler.classes[class_id][
                "name"] = self.name_input.current_state()
            if delete:
                self.delete_class(class_id)
            else:
                self.updateTab()

        dialog.setWindowTitle("Edit Class:")
        delete_button = QPushButton("Delete Class")
        finish_button = QPushButton("Done")
        delete_button.clicked.connect(lambda x: finish_func(delete=True))
        finish_button.clicked.connect(lambda x: finish_func())
        horiz_button_layout = QHBoxLayout()
        horiz_button_layout.addWidget(delete_button)
        horiz_button_layout.addWidget(finish_button)

        dialog.layout.addLayout(horiz_button_layout)

        dialog.setLayout(dialog.layout)
        dialog.show()

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

    def update_current_roi_selected(self):
        self.roi_label.setText(
            "ROI #{}:".format(self.roi_list_module.current_selected_roi))
        self.time_plot.clear()
        if self.roi_list_module.current_selected_roi is None:
            self.stat_labels["Area"].setText("Area: None")
            self.stat_labels["Eccentricity"].setText("Eccentricity: None")

            self.stat_labels["Extent"].setText("Extent: None")
            self.stat_labels["PSNR"].setText("PSNR: None")
            return
        try:

            roi_index = self.roi_list_module.current_selected_roi
            self.selectRoiTime(roi_index)
            roi_pix = self.data_handler.rois[roi_index]
            original_zeros = np.zeros(self.data_handler.shape, dtype=int).reshape(
                (-1, 1))
            original_zeros[roi_pix] = 1
            original_zeros = original_zeros.reshape(self.data_handler.shape)
            props = measure.regionprops(original_zeros)[0]
            extent = len(roi_pix) / ((self.data_handler.roi_max_cord_list[roi_index][
                                          0] -
                                      self.data_handler.roi_min_cord_list[roi_index][
                                          0]) * (self.data_handler.roi_max_cord_list[
                                                     roi_index][1] -
                                                 self.data_handler.roi_min_cord_list[
                                                     roi_index][1]))
            time_trace = self.data_handler.get_time_trace(roi_index)
            psnr = trial_PSNR(time_trace.reshape((1, -1)))[0]
            self.stat_labels["Area"].setText("Area: {}".format(len(roi_pix)))
            self.stat_labels["Eccentricity"].setText(
                "Eccentricity: {:.2f}".format(props["eccentricity"]))
            self.stat_labels["Extent"].setText("Extent: {:.2f}".format(extent))
            self.stat_labels["PSNR"].setText("PSNR: {:.2f}".format(psnr))



        except:
            # print("SOmething happened")
            pass

    def keyPressAction(self, event):

        if event.key() == 16777234:
            self.roi_list_module.select_roi_next(False)
            event.accept()
        if event.key() == 16777236:
            self.roi_list_module.select_roi_next(True)
            event.accept()

        return event.isAccepted()

    def updateTab(self
                  ):
        if (self.main_widget.checkThreadRunning()):

            self._time_trace_trial_select_list.set_items_from_list(
                self.data_handler.trials_all,
                self.data_handler.trials_loaded_time_trace_indices)
            if self.data_handler.rois_loaded:
                self.data_handler.gen_class_display_variables()
                self.roi_list_module.set_list_items(
                    self.main_widget.data_handler.rois_dict)
            self.set_class_buttons()
            self.roi_label.setText(
                "ROI #{}:".format(self.roi_list_module.current_selected_roi))
            self.image_view.reset_view()
