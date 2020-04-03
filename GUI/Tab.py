from PySide2.QtWidgets import *
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
    def __init__(self, name, column_1: List[QFrame], column_2: List[QFrame], column_2_display=True):
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
        file_open_button.setText("Load")
        file_open_button.clicked.connect(lambda: load_new_dataset(main_widget,dataset_file_input,save_dir_new_file))
        folder_open_button = QPushButton()
        folder_open_button.setText("Load")
        folder_open_button.clicked.connect(
            lambda: load_new_dataset(main_widget, dataset_folder_input, save_dir_new_folder))
        prev_session_open_button = QPushButton()
        prev_session_open_button.setText("Load")
        prev_session_open_button.clicked.connect(
            lambda: load_prev_session(main_widget, save_dir_load))
        file_open = Tab("File Open", column_2=[], column_2_display=False,column_1=[dataset_file_input,save_dir_new_file, file_open_button])
        folder_open = Tab("Folder Open", column_2=[], column_2_display=False,column_1=[dataset_folder_input,save_dir_new_folder,folder_open_button]
                                               )
        prev_session_open = Tab("Previous Session Open", column_2=[], column_2_display=False,column_1=[save_dir_load,prev_session_open_button])
        self.tab_selector = QTabWidget()
        self.tab_selector.addTab(file_open, file_open.name)
        self.tab_selector.addTab(folder_open, folder_open.name)
        self.tab_selector.addTab(prev_session_open, prev_session_open.name)

        super().__init__("FileOpenTab", column_1=[self.tab_selector], column_2=[], column_2_display=False)