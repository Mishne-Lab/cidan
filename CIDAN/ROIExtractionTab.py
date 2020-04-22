from PySide2 import QtCore

from CIDAN.Tab import Tab
from PySide2.QtWidgets import *
from PySide2.QtCore import *
import numpy as np
import pyqtgraph as pg
from CIDAN.DataHandlerWrapper import ROIExtractionThread
from CIDAN.SettingsModule import roi_extraction_settings
from CIDAN.ROIListModule import ROIListModule
from CIDAN.Input import OptionInput
class ROIExtractionTab(Tab):
    """Class controlling the ROI Extraction tab, inherits from Tab


        Attributes
        ----------
        main_widget : MainWidget
            A reference to the main widget
        data_handler : DataHandler
            A reference to the main DataHandler of MainWidget
        click_event : bool
            A bool that keeps track of whether a click event is currently happening used
            by roi_click_event and select_roi
        time_plot : pg.PlotWidget
            the plot for the time traces
        roi_list_module : ROIListModule
            The module the controlls the list of ROIs
        thread : ROIExtractionThread
            The thread that runs the roi extraction process
        background_slider : QSlider
            slider that determines intensity of background image+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        """
    def __init__(self,main_widget):

        self.main_widget = main_widget
        self.data_handler = main_widget.data_handler
        self.click_event = False
        self.add_image = False
        self.outlines = True
        # overload the image click and drag properites on the image item
        self.image_item = self.main_widget.roi_image_view.image_view.getImageItem()
        self.image_item.mouseClickEvent = lambda x: self.roi_view_click(x)
        self.image_item.mouseDragEvent = lambda x: self.roi_view_drag(x)
        # This part creates the top left settings/roi list view in two tabs
        tab_selector_roi = QTabWidget()
        tab_selector_roi.setStyleSheet("QTabWidget {font-size: 20px;}")
        # This is the second tab
        roi_modification_tab = QWidget()
        roi_modification_tab.setStyleSheet("margin:0px; padding: 0px;")

        roi_modification_tab_layout = QVBoxLayout()
        roi_modification_tab.setLayout(roi_modification_tab_layout)
        self.roi_list_module = ROIListModule(main_widget.data_handler, self)
        roi_modification_tab_layout.addWidget(self.roi_list_module)
        roi_modification_button_top_layout = QHBoxLayout()
        roi_modification_tab_layout.addLayout(roi_modification_button_top_layout)
        add_to_roi = QPushButton(text="Add to ROI")
        subtract_from_roi = QPushButton(text="Subtract from ROI")

        roi_modification_button_top_layout.addWidget(add_to_roi)
        roi_modification_button_top_layout.addWidget(subtract_from_roi)
        painter_options = OptionInput("Pixel Selector Brush:", "",
                                      lambda x,y: self.setSelectorBrushType(y),
                                      default_index=0, tool_tip="",
                                      val_list=["Off", "Add to Selection",
                                                "Subtract from Selection"])
        roi_modification_tab_layout.addWidget(painter_options)






        # Start of first tab with the process button
        process_button = QPushButton()
        process_button.setText("Apply Settings")
        self.thread = ROIExtractionThread(main_widget, process_button, self.roi_list_module,self)
        self.main_widget.thread_list.append(self.thread)
        process_button.clicked.connect(lambda: self.thread.runThread())
        self.roi_settings = QWidget()
        self.roi_settings_layout = QVBoxLayout()
        self.roi_settings.setLayout(self.roi_settings_layout)
        self.roi_settings_layout.addWidget(roi_extraction_settings(main_widget))
        self.roi_settings_layout.addWidget(process_button)
        # adding the tabs to the
        tab_selector_roi.addTab(self.roi_settings, "ROI Creation")
        tab_selector_roi.addTab(roi_modification_tab, "ROI Modification")

        # Initialization of the background and rois
        self.current_background_intensity = 1
        self.set_background("", "Max Image", update_image=False)
        if self.main_widget.data_handler.rois_loaded:
            self.thread.endThread(True)



        # Image and time trace settings window


        display_settings_layout = QVBoxLayout()

        display_settings = QWidget()
        display_settings.setLayout(display_settings_layout)
        image_chooser = OptionInput("ROI Display type::", "",
                                         on_change_function=self.set_image,
                                         default_index=0,
                                         tool_tip="Choose background to display",
                                         val_list=["Outlines", "Blob"])

        display_settings_layout.addWidget(image_chooser)






        background_chooser = OptionInput("Background:", "",
                                              on_change_function=self.set_background, default_index=2,
                                              tool_tip="Choose background to display",
                                              val_list=["Blank Image", "Mean Image", "Max Image", "Temporal Correlation Image", "Eigen Norm Image"])

        display_settings_layout.addWidget(background_chooser)
        background_slider_layout = QHBoxLayout()
        background_slider_layout.addWidget(QLabel("0"))
        self.background_slider = QSlider(Qt.Horizontal)
        self.background_slider.setMinimum(0)
        self.background_slider.setValue(10)
        self.background_slider.setMaximum(100)
        self.background_slider.setSingleStep(1)
        self.background_slider.valueChanged.connect(self.intensity_slider_changed)
        background_slider_layout.addWidget(self.background_slider)
        background_slider_layout.addWidget(QLabel("10"))

        display_settings_layout.addWidget(QLabel("Change background intensity:"))
        display_settings_layout.addLayout(background_slider_layout)

        # Initialize the tab with each column

        # this is to override how we do the column 2 to replace it with a split view
        tab_selector_time_trace = QTabWidget()
        tab_selector_time_trace.setStyleSheet("QTabWidget {font-size: 20px;}")
        tab_selector_time_trace.setMaximumHeight(200)
        self.time_plot = pg.PlotWidget()
        self.time_plot.showGrid(x=True, y=True, alpha=0.3)
        tab_selector_time_trace.addTab(self.time_plot, "Time Trace Plot")
        tab_selector_time_trace.addTab(QWidget(), "Time Trace Settings")


        # part about selecting pixels
        self.select_pixel_on = True # whether you can currently select pixels in the image
        self.current_selected_pixels= [] # list of currently selected pixels in their 1d number format
        self.select_mode = "add" # possibilities add and subtract from current selection
        roi_view_tabs = QTabWidget()
        roi_view_tabs.setStyleSheet("QTabWidget {font-size: 20px;}")
        # roi_view_tabs.setStyleSheet(
        #     "QButton, QLabel, QSlider {padding: 5px; margin: 5px;}")
        # roi_view_tabs.setStyleSheet("QTabWidget {font-size: 20px;}")
        self.main_widget.roi_image_view.setStyleSheet("margin:0px; border:0px  solid rgb(50, 65, "
                           "75); padding: 0px;")
        roi_view_tabs.addTab(self.main_widget.roi_image_view, "ROI Display")
        roi_view_tabs.addTab(display_settings, "Display Settings")
        self.column_2 = [roi_view_tabs, tab_selector_time_trace ]
        super().__init__("ROI Extraction",
                         column_1=[tab_selector_roi],
                         column_2=self.column_2, column_2_display=True)
        # self.setStyleSheet("SettingsModule { border:1px solid rgb(50, 65, "
        #                    "75);} ")
        # column_2_split = QSplitter(Qt.Vertical)  # Layout for column 2 with split part
        # for module in self.column_2:
        #     column_2_split.addWidget(module)
        # column_2_split.setSizes([400, 100])
        #
        #
        # column_2_layout_box.addWidget(column_2_split)
        # # column_2_layout_box.addWidget(tab_selector_image)
        # self.layout.addLayout(column_2_layout_box)
    def setSelectorBrushType(self,type):
        if type == "Off":
            self.select_pixel_on = False
        else:
            self.select_pixel_on = True
            self.select_mode = "add" if type=="Add to Selection" else "subtract"
    def draw(self,pos):
        pass
    def intensity_slider_changed(self):
        self.current_background_intensity= float(self.background_slider.value()) / 10
        self.updateImageDisplay()
    def set_background(self, name,func_name, update_image=True):
        # Background refers to the image behind the rois
        shape = self.main_widget.data_handler.shape
        if func_name == "Mean Image":
            self.current_background = self.main_widget.data_handler.mean_image.reshape([-1, 1])
        if func_name == "Max Image":
            self.current_background = self.main_widget.data_handler.max_image.reshape([-1, 1])
        if func_name == "Blank Image":
            self.current_background = np.zeros([shape[1]*shape[2],1])
        if func_name == "Temporal Correlation Image":
            self.current_background = self.data_handler.temporal_correlation_image.reshape([-1, 1])
        if func_name == "Eigen Norm Image":
            self.current_background = self.data_handler.eigen_norm_image.reshape([-1, 1])

        if update_image:
            self.updateImageDisplay()

    def set_image(self, name, func_name, update_image=True):
        # Background refers to the image behind the rois
        shape = self.main_widget.data_handler.edge_roi_image_flat.shape
        if func_name == "Outlines":
            self.outlines=True
            self.add_image = False
            self.roi_image_flat = np.hstack([self.data_handler.edge_roi_image_flat,
                                             np.zeros(shape),
                                             np.zeros(shape)])
        if func_name == "Blob":
            self.outlines = False
            self.add_image =True
            self.roi_image_flat = self.main_widget.data_handler.pixel_with_rois_color_flat

        if update_image:
            self.updateImageDisplay()
    def updateImageDisplay(self, new=False):
        # new is to determine whether the zoom should be saved
        # TODO add in update with image paint layer
        shape = self.main_widget.data_handler.dataset_filtered.shape
        if not hasattr(self, "select_image_flat"):
            self.select_image_flat = np.zeros([shape[1]*shape[2], 3])
        range_list=self.main_widget.roi_image_view.image_view.view.viewRange()
        print(range_list)
        shape = self.main_widget.data_handler.dataset_filtered.shape
        background_max = self.current_background.max()
        background_image_scaled = (self.current_background_intensity * 255 / (background_max if background_max!= 0 else 1)) * self.current_background
        background_image_scaled_3_channel = np.hstack([background_image_scaled,background_image_scaled ,background_image_scaled])
        if self.add_image:
            combined = self.roi_image_flat + background_image_scaled_3_channel+self.select_image_flat
        else:
            combined = background_image_scaled + self.select_image_flat
            mask = np.any(self.roi_image_flat != [0, 0, 0], axis=1)
            combined[mask] = self.roi_image_flat[mask]

        combine_reshaped = combined.reshape((shape[1], shape[2], 3))
        self.main_widget.roi_image_view.setImage(combine_reshaped)
        if not new:
            self.main_widget.roi_image_view.image_view.view.setRange(xRange=range_list[0],
                                                                 yRange=range_list[1])
            range_list = self.main_widget.roi_image_view.image_view.view.viewRange()
            print(range_list)

        pass

    def selectRoi(self, num, ):

        color_select = (245, 249, 22)
        color_roi = self.main_widget.data_handler.color_list[(num-1) % len(self.main_widget.data_handler.color_list)]
        shape = self.main_widget.data_handler.dataset_filtered.shape
        self.select_image_flat[self.main_widget.data_handler.clusters[num - 1]] = color_select
        self.updateImageDisplay()
        if self.click_event:
            self.click_event = False
        else:

            max_cord_list = np.array([x for num,x in enumerate(self.main_widget.data_handler.cluster_max_cord_list) if self.roi_list_module.roi_check_list[num]])
            max_cord = list(np.max(max_cord_list, axis=0))
            min_cord_list =np.array([x for num,x in enumerate(self.main_widget.data_handler.cluster_min_cord_list) if self.roi_list_module.roi_check_list[num]])
            min_cord = list( np.min(min_cord_list, axis=0))

            self.main_widget.roi_image_view.image_view.getView().setXRange(min_cord[1], max_cord[1])
            self.main_widget.roi_image_view.image_view.getView().setYRange(min_cord[0],
                                                                       max_cord[0])
        pen = pg.mkPen(color=color_roi, width=3)
        self.time_plot.plot(self.main_widget.data_handler.get_time_trace(num), pen=pen)
        self.time_plot.enableAutoRange(axis=0)
    def deselectRoi(self, num):

        color = self.main_widget.data_handler.color_list[(num-1) % len(self.main_widget.data_handler.color_list)]
        shape = self.main_widget.data_handler.dataset_filtered.shape
        shape_flat = self.data_handler.edge_roi_image_flat.shape
        self.select_image_flat[self.main_widget.data_handler.clusters[num - 1]] = color if not self.outlines \
            else np.hstack([self.data_handler.edge_roi_image_flat,
                                                np.zeros(shape_flat),
                                                np.zeros(shape_flat)])[self.main_widget.data_handler.clusters[num - 1]]
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
        if event.button() == QtCore.Qt.RightButton:
            if self.image_item.raiseContextMenu(event):
                event.accept()
        event.accept()
        pos = event.pos()

        x = int(pos.x())
        y = int(pos.y())
        if self.select_pixel_on:
            if self.select_mode == "add":
                self.image_item.image[x,y]=[0,255,0]
                self.image_item.updateImage()
        else:
            self.click_event = True
            pixel_with_rois_flat = self.main_widget.data_handler.pixel_with_rois_flat
            shape = self.main_widget.data_handler.dataset_filtered.shape
            roi_num = int(pixel_with_rois_flat[shape[2] * x + y])
            # TODO change to int
            if roi_num != 0:
                self.roi_list_module.set_current_select(roi_num)


    def roi_view_drag(self,event):
        # if event.button() == QtCore.Qt.RightButton:
        #     if self.image_item.raiseContextMenu(event):
        #         event.accept()
        event.accept()
        pos = event.pos()

        x = int(pos.x())
        y = int(pos.y())
        if self.select_pixel_on:
            if self.select_mode == "add":
                self.image_item.image[x, y] = [0, 255, 0]
                self.image_item.updateImage()
    def pixel_paint(self,x,y):
        pass # TODO use slicing to update pixel based on current thing
    def check_pos_in_image(self, x,y):
        pass
        # TODO add in way to check if in image