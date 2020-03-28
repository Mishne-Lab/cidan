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
            self.column_1_layout.addWidget(module, stretch = module.importance)
        self.layout.addLayout(self.column_1_layout, stretch=1)
        if column_2_display:
            self.column_2_layout = QVBoxLayout() # Layout for column 2
            for module in column_2:
                self.column_2_layout.addWidget(module, strech=module.importance)
            self.layout.addLayout(self.column_2_layout, stretch=1)
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
        super().__init__("Preprocessing", column_1=[preprocessing_settings(main_widget), self.process_button], column_2=[
                main_widget.preprocess_image_view])



class ROIExtractionTab(Tab):
    def __init__(self,main_widget):

        self.main_widget = main_widget
        self.process_button = QPushButton()
        self.process_button.importance = 1
        self.process_button.setText("Apply Settings")
        self.roi_list_module = ROIListModule(main_widget.data_handler, self)
        self.thread = ROIExtractionThread(main_widget, self.process_button, self.roi_list_module,self)
        self.main_widget.thread_list.append(self.thread)
        self.current_image_flat = None
        self.process_button.clicked.connect(lambda: self.thread.runThread())
        self.main_widget.roi_image_view.image_view.getImageItem().mouseClickEvent = lambda x: roi_view_click(self.main_widget,self.roi_list_module, x)
        self.tab_selector = QTabWidget()
        self.tab_selector.setMaximumWidth(400)
        self.tab_selector.setStyleSheet("QTabWidget {font-size: 20px;}")
        self.tab_selector.importance = 1
        self.tab_selector.addTab(Tab("ROI Settings", column_1=[roi_extraction_settings(main_widget), self.process_button], column_2=[],column_2_display=False)," ROI Settings")
        self.tab_selector.addTab(self.roi_list_module,"ROI List")
        if self.main_widget.data_handler.rois_loaded:
            self.thread.endThread()
        self.time_plot = pg.plot()
        self.time_plot.importance =.1
        self.time_plot.showGrid(x = True, y = True, alpha = 0.3)

        super().__init__("ROI Extraction", column_1=[self.tab_selector],
                             column_2=[], column_2_display=False)
        # this is to override how we do the column 2 to replace it with a splitter
        self.column_2 = [self.main_widget.roi_image_view,self.time_plot]
        self.column_2_layout = QSplitter(Qt.Vertical)  # Layout for column 2
        for module in self.column_2:
            self.column_2_layout.addWidget(module)
        self.column_2_layout.setSizes((300,500))

        self.layout.addWidget(self.column_2_layout)

    def select_roi(self, num):
        color_select = (245, 249, 22)
        color_roi = self.main_widget.data_handler.color_list[(num-1) % len(self.main_widget.data_handler.color_list)]
        shape = self.main_widget.data_handler.dataset.shape
        self.current_image_flat[self.main_widget.data_handler.clusters[num-1]] = color_select
        self.main_widget.roi_image_view.setImage(self.current_image_flat.reshape((shape[1],shape[2],3)))
        pen = pg.mkPen(color=color_roi, width=3)
        self.time_plot.plot(self.main_widget.data_handler.calculate_time_trace(num), pen=pen)

    def unselect_roi(self,num):

        color = self.main_widget.data_handler.color_list[(num-1) % len(self.main_widget.data_handler.color_list)]
        shape = self.main_widget.data_handler.dataset.shape
        self.current_image_flat[self.main_widget.data_handler.clusters[num - 1]] = color
        self.main_widget.roi_image_view.setImage(self.current_image_flat.reshape((shape[1],shape[2],3)))
        self.time_plot.clear()
        for num2, x in zip(range(1,len(self.roi_list_module.roi_check_list)),self.roi_list_module.roi_check_list):
            if x:
                color_roi = self.main_widget.data_handler.color_list[
                    (num2 - 1) % len(self.main_widget.data_handler.color_list)]

                pen = pg.mkPen(color=color_roi, width=3)
                self.time_plot.plot(
                    self.main_widget.data_handler.calculate_time_trace(num2), pen=pen)

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