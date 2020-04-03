from GUI.Tab import Tab
from PySide2.QtWidgets import *
from PySide2.QtCore import *
import numpy as np
import pyqtgraph as pg
from GUI.DataHandlerWrapper import ROIExtractionThread
from GUI.SettingsModule import roi_extraction_settings
from GUI.ROIListModule import ROIListModule
from GUI.Input import OptionInput
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
    def __init__(self,main_widget):

        self.main_widget = main_widget
        self.data_handler = main_widget.data_handler
        # This part creates the top left settings/roi list view in two tabs
        self.tab_selector_roi = QTabWidget()
        self.tab_selector_roi.setMaximumWidth(450)
        self.tab_selector_roi.setStyleSheet("QTabWidget {font-size: 20px;}")
        # This is the second tab
        self.roi_list_module = ROIListModule(main_widget.data_handler, self)
        # Start of first tab with the process button
        process_button = QPushButton()
        process_button.setText("Apply Settings")
        self.thread = ROIExtractionThread(main_widget, process_button, self.roi_list_module,self)
        self.main_widget.thread_list.append(self.thread)
        process_button.clicked.connect(lambda: self.thread.runThread())
        self.main_widget.roi_image_view.image_view.getImageItem().mouseClickEvent = lambda x: self.roi_view_click(x)
        self.roi_settings = QWidget()
        self.roi_settings_layout = QVBoxLayout()
        self.roi_settings.setLayout(self.roi_settings_layout)
        self.roi_settings_layout.addWidget(roi_extraction_settings(main_widget))
        self.roi_settings_layout.addWidget(process_button)
        # adding the tabs to the
        self.tab_selector_roi.addTab(self.roi_settings, "ROI Settings")
        self.tab_selector_roi.addTab(self.roi_list_module, "ROI List")

        # Initialization of the background and rois
        self.current_background_intensity = 1
        self.set_background("", "Blank Image", update_image=False)
        if self.main_widget.data_handler.rois_loaded:
            self.thread.endThread(True)

        self.time_plot = pg.PlotWidget()
        self.time_plot.showGrid(x = True, y = True, alpha = 0.3)

        # Image and time trace settings window
        self.tab_selector_image = QTabWidget()
        self.tab_selector_image.setMaximumHeight(200)
        self.tab_selector_image.setStyleSheet("QButton, QLabel, QSlider {padding: 5px; margin: 5px;}")
        self.tab_selector_image.setStyleSheet("QTabWidget {font-size: 20px;}")
        self.background_settings_layout = QVBoxLayout()

        self.background_settings = QWidget()
        self.background_settings.setMaximumHeight(150)
        self.background_settings.setLayout(self.background_settings_layout)
        self.background_chooser = OptionInput("Background:", "",
                                              on_change_function=self.set_background, default_index=0,
                                              tool_tip="Choose background to display",
                                              val_list=["Blank Image", "Mean Image", "Max Image", "Temporal Correlation Image", "Eigennorm image"])

        self.background_settings_layout.addWidget(self.background_chooser)
        self.background_slider_layout = QHBoxLayout()
        self.background_slider_layout.addWidget(QLabel("0"))
        self.background_slider = QSlider(Qt.Horizontal)
        self.background_slider.setMinimum(0)
        self.background_slider.setValue(10)
        self.background_slider.setMaximum(100)
        self.background_slider.setSingleStep(1)
        self.background_slider.valueChanged.connect(self.intensity_slider_changed)
        self.background_slider_layout.addWidget(self.background_slider)
        self.background_slider_layout.addWidget(QLabel("10"))

        self.background_settings_layout.addWidget(QLabel("Change background intensity:"))
        self.background_settings_layout.addLayout(self.background_slider_layout)
        self.tab_selector_image.addTab(self.background_settings,"Background Settings")
        self.tab_selector_image.addTab(QWidget(), "Time Trace Settings")

        # Initialize the tab with each column
        super().__init__("ROI Extraction", column_1=[self.tab_selector_roi, self.tab_selector_image],
                         column_2=[], column_2_display=False)
        # this is to override how we do the column 2 to replace it with a splitter
        self.column_2_layout_box = QVBoxLayout()
        self.time_plot_layout_wrapper = QWidget()
        self.time_plot_layout = QVBoxLayout()
        self.time_plot_layout_wrapper.setLayout(self.time_plot_layout)
        self.time_plot_layout.addWidget(QLabel("Time Trace Plot:"))
        self.time_plot_layout.addWidget(self.time_plot)

        self.column_2 = [self.main_widget.roi_image_view, self.time_plot_layout_wrapper]


        self.column_2_split = QSplitter(Qt.Vertical)  # Layout for column 2
        for module in self.column_2:
            self.column_2_split.addWidget(module)
        self.column_2_split.setSizes([400, 100])


        self.column_2_layout_box.addWidget(self.column_2_split)
        # self.column_2_layout_box.addWidget(self.tab_selector_image)
        self.layout.addLayout(self.column_2_layout_box)
    def intensity_slider_changed(self):
        self.current_background_intensity= float(self.background_slider.value()) / 10
        self.updateImageDisplay()
    def set_background(self, name,func_name, update_image=True):
        # Background refers to the image behind the rois
        shape = self.main_widget.data_handler.dataset.shape
        if func_name == "Mean Image":
            self.current_background = self.main_widget.data_handler.mean_image.reshape([-1, 1])
        if func_name == "Max Image":
            self.current_background = self.main_widget.data_handler.max_image.reshape([-1, 1])
        if func_name == "Blank Image":
            self.current_background = np.zeros([shape[1]*shape[2],1])
        if func_name == "Temporal Correlation Image":
            self.current_background = self.data_handler.temporal_correlation_image.reshape([-1, 1])

        if update_image:
            self.updateImageDisplay()
    def updateImageDisplay(self):
        shape = self.main_widget.data_handler.dataset.shape
        background_max =self.current_background.max()
        background_image_scaled = (
                                 self.current_background_intensity * 255 / (background_max if background_max!= 0 else 1)) * self.current_background

        combined = self.current_image_flat + background_image_scaled
        combine_reshaped = combined.reshape((shape[1], shape[2], 3))
        self.main_widget.roi_image_view.setImage(combine_reshaped)
    def displayBlankImageBackground(self):
        shape = self.main_widget.data_handler.dataset.shape

        self.main_widget.roi_image_view.setImage(self.current_image_flat.reshape((shape[1],shape[2],3)))
    def displayMeanImageBackground(self):
        # TODO add slider for background intensity
        shape = self.main_widget.data_handler.dataset.shape

        mean_image = self.main_widget.data_handler.mean_image.reshape([-1,1])
        mean_image = (self.current_background_intensity * 255 / mean_image.max()) * mean_image

        combined = self.current_image_flat+ mean_image
        combine_reshaped = combined.reshape((shape[1],shape[2],3))
        self.main_widget.roi_image_view.setImage(combine_reshaped)
    def displayMaxImageBackground(self):
        # TODO add slider for background intensity
        shape = self.main_widget.data_handler.dataset.shape

        max_image = self.main_widget.data_handler.max_image.reshape([-1,1])
        max_image = (self.current_background_intensity * 255 / max_image.max()) * max_image

        combined = self.current_image_flat + max_image
        combine_reshaped = combined.reshape((shape[1],shape[2],3))
        self.main_widget.roi_image_view.setImage(combine_reshaped)
    def selectRoi(self, num):
        color_select = (245, 249, 22)
        color_roi = self.main_widget.data_handler.color_list[(num-1) % len(self.main_widget.data_handler.color_list)]
        shape = self.main_widget.data_handler.dataset.shape
        self.current_image_flat[self.main_widget.data_handler.clusters[num-1]] = color_select
        self.updateImageDisplay()
        pen = pg.mkPen(color=color_roi, width=3)
        self.time_plot.plot(self.main_widget.data_handler.get_time_trace(num), pen=pen)
        self.time_plot.enableAutoRange(axis=0)
    def deselectRoi(self, num):

        color = self.main_widget.data_handler.color_list[(num-1) % len(self.main_widget.data_handler.color_list)]
        shape = self.main_widget.data_handler.dataset.shape
        self.current_image_flat[self.main_widget.data_handler.clusters[num - 1]] = color
        self.updateImageDisplay()
        self.time_plot.clear()
        self.time_plot.enableAutoRange(axis=0)
        for num2, x in zip(range(1,len(self.roi_list_module.roi_check_list)),self.roi_list_module.roi_check_list):
            if x:
                color_roi = self.main_widget.data_handler.color_list[
                    (num2 - 1) % len(self.main_widget.data_handler.color_list)]

                pen = pg.mkPen(color=color_roi, width=3)
                self.time_plot.plot(
                    self.main_widget.data_handler.get_time_trace(num2), pen=pen)

    def roi_view_click(self, event):
        event.accept()
        pos = event.pos()
        x = int(pos.x())
        y = int(pos.y())
        pixel_with_rois_flat = self.main_widget.data_handler.pixel_with_rois_flat
        shape = self.main_widget.data_handler.dataset.shape
        roi_num = int(pixel_with_rois_flat[shape[2] * x + y])
        # TODO change to int
        if roi_num != 0:
            self.roi_list.set_current_select(roi_num)
