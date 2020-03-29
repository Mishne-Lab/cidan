from PySide2.QtWidgets import *
from GUI.Module import Module
from typing import Union, Any, List, Optional, cast, Tuple, Dict
from GUI.SettingsModule import *
from GUI.roiTools import *
from GUI.ImageViewModule import ImageViewModule
from GUI.Input import FileInput
from GUI.DataHandlerWrapper import *
from GUI.fileHandling import *
import pyqtgraph as pg
from GUI.ROIListModule import *
class Tab(QWidget):
    def __init__(self, name, column_1: List[Module], column_2: List[Module], column_2_display=True):
        super().__init__()
        self.name = name
        self.column_1 = column_1
        self.column_2 = column_2
        self.setMinimumHeight(500)
        self.layout = QHBoxLayout() # Main layout class
        self.column_1_layout = QVBoxLayout() # Layout for column 1
        for module in column_1:
            self.column_1_layout.addWidget(module)
        self.layout.addLayout(self.column_1_layout, stretch=1)
        if column_2_display:
            self.column_2_split = QVBoxLayout() # Layout for column 2
            for module in column_2:
                self.column_2_split.addWidget(module)
            self.layout.addLayout(self.column_2_split, stretch=1)
        self.setLayout(self.layout)
class PreprocessingTab(Tab):
    def __init__(self, main_widget):
        self.main_widget = main_widget
        self.data_handler = self.main_widget.data_handler
        self.process_button = QPushButton()
        self.process_button.importance =1
        self.process_button.setText("Apply Settings")
        thread = PreprocessThread(main_widget,self.process_button)
        main_widget.thread_list.append(thread)
        self.process_button.clicked.connect(lambda: thread.runThread())
        self.main_widget.preprocess_image_view.setImage(
            self.data_handler.calculate_filters())
        self.image_buttons = QWidget()
        self.image_buttons_layout = QHBoxLayout()
        self.image_buttons.setLayout(self.image_buttons_layout)
        self.max_image_button = QPushButton()
        self.max_image_button.setText("Max Image")
        self.max_image_button.clicked.connect(lambda: self.main_widget.preprocess_image_view.setImage(
            self.data_handler.max_image))
        self.stack_button = QPushButton()
        self.stack_button.setText("Filtered Stack")
        self.stack_button.clicked.connect(
            lambda: self.main_widget.preprocess_image_view.setImage(
                self.data_handler.dataset_filtered))
        self.orig_stack_button = QPushButton()
        self.orig_stack_button.setText("Original Stack")
        self.orig_stack_button.clicked.connect(
            lambda: self.main_widget.preprocess_image_view.setImage(
                self.data_handler.dataset))
        self.mean_image_button = QPushButton()
        self.mean_image_button.setText("Mean Image")
        self.mean_image_button.clicked.connect(
            lambda: self.main_widget.preprocess_image_view.setImage(
                self.data_handler.mean_image))
        self.image_buttons_layout.addWidget(self.orig_stack_button)
        self.image_buttons_layout.addWidget(self.stack_button)
        self.image_buttons_layout.addWidget(self.max_image_button)
        self.image_buttons_layout.addWidget(self.mean_image_button)
        super().__init__("Preprocessing", column_1=[preprocessing_settings(main_widget), self.process_button], column_2=[
                main_widget.preprocess_image_view,self.image_buttons])



class ROIExtractionTab(Tab):
    def __init__(self,main_widget):

        self.main_widget = main_widget
        self.current_background_func = self.displayBlankImageBackground
        self.process_button = QPushButton()
        self.process_button.importance = 1
        self.process_button.setText("Apply Settings")
        self.roi_list_module = ROIListModule(main_widget.data_handler, self)
        self.thread = ROIExtractionThread(main_widget, self.process_button, self.roi_list_module,self)
        self.main_widget.thread_list.append(self.thread)
        shape = self.main_widget.data_handler.dataset.shape
        self.process_button.clicked.connect(lambda: self.thread.runThread())
        self.main_widget.roi_image_view.image_view.getImageItem().mouseClickEvent = lambda x: roi_view_click(self.main_widget,self.roi_list_module, x)
        self.roi_settings = QWidget()
        self.roi_settings_layout = QVBoxLayout()
        self.roi_settings.setLayout(self.roi_settings_layout)
        self.roi_settings_layout.addWidget(roi_extraction_settings(main_widget))
        self.roi_settings_layout.addWidget(self.process_button)
        self.tab_selector_roi = QTabWidget()
        self.tab_selector_roi.setMaximumWidth(450)
        self.tab_selector_roi.setStyleSheet("QTabWidget {font-size: 20px;}")
        self.tab_selector_roi.importance = 1
        self.tab_selector_roi.addTab(self.roi_settings, "ROI Settings")
        self.tab_selector_roi.addTab(self.roi_list_module, "ROI List")
        if self.main_widget.data_handler.rois_loaded:
            self.thread.endThread(True)
        self.time_plot = pg.PlotWidget()
        self.time_plot.importance =.1
        self.time_plot.showGrid(x = True, y = True, alpha = 0.3)


        self.tab_selector_image = QTabWidget()
        self.tab_selector_image.setMaximumHeight(200)
        self.tab_selector_image.setStyleSheet("QButton, QLabel, QSlider {padding: 5px; margin: 5px;}")
        # self.tab_selector_image.setMaximumWidth(450)
        self.tab_selector_image.setStyleSheet("QTabWidget {font-size: 20px;}")
        self.tab_selector_image.importance = 1
        self.background_settings_layout = QVBoxLayout()

        self.background_settings = QWidget()
        self.background_settings.setMaximumHeight(150)
        self.background_settings.setLayout(self.background_settings_layout)
        self.background_image_buttons_layout = QHBoxLayout()
        self.blank_image_button = QPushButton()
        self.blank_image_button.setText("Blank Background")

        self.blank_image_button.clicked.connect(
            lambda: self.set_background(self.displayBlankImageBackground))
        self.background_image_buttons_layout.addWidget(self.blank_image_button)

        self.mean_image_button = QPushButton()

        self.mean_image_button.setText("Mean Image")
        self.current_background_func = self.displayBlankImageBackground
        self.mean_image_button.clicked.connect(
            lambda: self.set_background(self.displayMeanImageBackground))
        self.background_image_buttons_layout.addWidget(self.mean_image_button)
        self.max_image_button = QPushButton()
        self.max_image_button.setText("Max Image")
        self.max_image_button.clicked.connect(
            lambda: self.set_background(self.displayMaxImageBackground))
        self.background_image_buttons_layout.addWidget(self.max_image_button)
        self.current_background_intensity = 1
        self.background_slider_layout = QHBoxLayout()
        self.background_slider_layout.addWidget(QLabel("0"))
        self.background_slider = QSlider(Qt.Horizontal)
        self.background_slider.setMinimum(0)
        self.background_slider.setValue(10)
        self.background_slider.setMaximum(50)
        self.background_slider.setSingleStep(1)
        self.background_slider.valueChanged.connect(self.intensity_slider_changed)
        self.background_slider_layout.addWidget(self.background_slider)
        self.background_slider_layout.addWidget(QLabel("10"))

        self.background_settings_layout.addWidget(QLabel("Change Background:"))
        self.background_settings_layout.addLayout(self.background_image_buttons_layout)
        self.background_settings_layout.addWidget(QLabel("Change background intensity:"))
        self.background_settings_layout.addLayout(self.background_slider_layout)
        self.tab_selector_image.addTab(self.background_settings,"Background Settings")
        self.tab_selector_image.addTab(QWidget(), "Time Trace Settings")
        super().__init__("ROI Extraction", column_1=[self.tab_selector_roi, self.tab_selector_image],
                         column_2=[], column_2_display=False)
        # this is to override how we do the column 2 to replace it with a splitter
        self.column_2_layout_box = QVBoxLayout()
        self.time_plot_layout_wrapper = QWidget()
        self.time_plot_layout = QVBoxLayout()
        self.time_plot_layout_wrapper.setLayout(self.time_plot_layout)
        self.time_plot_layout.addWidget(QLabel("Time Trace Plot:"))
        self.time_plot_layout.addWidget(self.time_plot)

        self.column_2 = [self.main_widget.roi_image_view,self.time_plot_layout_wrapper]


        self.column_2_split = QSplitter(Qt.Vertical)  # Layout for column 2
        for module in self.column_2:
            self.column_2_split.addWidget(module)
        self.column_2_split.setSizes([400, 100])


        self.column_2_layout_box.addWidget(self.column_2_split)
        # self.column_2_layout_box.addWidget(self.tab_selector_image)
        self.layout.addLayout(self.column_2_layout_box)
    def intensity_slider_changed(self):
        self.current_background_intensity= float(self.background_slider.value()) / 10
        self.current_background_func()
    def set_background(self, func):
        # Background refers to the image behind the rois
        self.current_background_func=func
        func()
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
        self.current_background_func()
        pen = pg.mkPen(color=color_roi, width=3)
        self.time_plot.plot(self.main_widget.data_handler.get_time_trace(num), pen=pen)
        self.time_plot.enableAutoRange(axis=0)
    def deselectRoi(self, num):

        color = self.main_widget.data_handler.color_list[(num-1) % len(self.main_widget.data_handler.color_list)]
        shape = self.main_widget.data_handler.dataset.shape
        self.current_image_flat[self.main_widget.data_handler.clusters[num - 1]] = color
        self.current_background_func()
        self.time_plot.clear()
        self.time_plot.enableAutoRange(axis=0)
        for num2, x in zip(range(1,len(self.roi_list_module.roi_check_list)),self.roi_list_module.roi_check_list):
            if x:
                color_roi = self.main_widget.data_handler.color_list[
                    (num2 - 1) % len(self.main_widget.data_handler.color_list)]

                pen = pg.mkPen(color=color_roi, width=3)
                self.time_plot.plot(
                    self.main_widget.data_handler.get_time_trace(num2), pen=pen)

class AnalysisTab(Tab):
    def __init__(self, main_widget):
        super().__init__("Analysis", column_1=[], column_2=[])
class FileOpenTab(Tab):
    def __init__(self,main_widget):
        # TODO Make this less ugly can reorganize code
        dataset_file_input = FileInput("Dataset File:", "", "","","Select a file to load in", isFolder=2, forOpen=True)
        dataset_folder_input = FileInput("Dataset Folder:", "", "","","Select a folder to load in", isFolder=1, forOpen=True)
        save_dir_new_file = FileInput("Save Directory Location:", "", "","","Select a place to save outputs", isFolder=1, forOpen=False)
        save_dir_new_folder = FileInput("Save Directory Location:", "", "","","Select a place to save outputs", isFolder=1, forOpen=False)

        save_dir_load = FileInput("Previous Session Location:", "", "", "",
                      "Select the save directory for a previous session", isFolder=1, forOpen=True)
        file_open_button = QPushButton()
        file_open_button.importance = 1
        file_open_button.setText("Load")
        file_open_button.clicked.connect(lambda: load_new_dataset(main_widget,dataset_file_input,save_dir_new_file))
        folder_open_button = QPushButton()
        folder_open_button.importance = 1
        folder_open_button.setText("Load")
        folder_open_button.clicked.connect(
            lambda: load_new_dataset(main_widget, dataset_folder_input, save_dir_new_folder))
        prev_session_open_button = QPushButton()
        prev_session_open_button.importance = 1
        prev_session_open_button.setText("Load")
        prev_session_open_button.clicked.connect(
            lambda: load_prev_session(main_widget, save_dir_load))
        file_open = Tab("File Open", column_2=[], column_2_display=False,column_1=[dataset_file_input,save_dir_new_file, file_open_button])
        folder_open = Tab("Folder Open", column_2=[], column_2_display=False,column_1=[dataset_folder_input,save_dir_new_folder,folder_open_button]
                                               )
        prev_session_open = Tab("Previous Session Open", column_2=[], column_2_display=False,column_1=[save_dir_load,prev_session_open_button])
        self.tab_selector = QTabWidget()
        self.tab_selector.importance = 1
        self.tab_selector.addTab(file_open, file_open.name)
        self.tab_selector.addTab(folder_open, folder_open.name)
        self.tab_selector.addTab(prev_session_open, prev_session_open.name)

        super().__init__("FileOpenTab", column_1=[self.tab_selector], column_2=[], column_2_display=False)