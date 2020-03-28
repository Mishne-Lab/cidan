from PySide2.QtWidgets import *
from GUI.Module import Module
from typing import Union, Any, List, Optional, cast, Tuple, Dict
from GUI.SettingsModule import *
from GUI.ImageViewModule import ImageViewModule
from GUI.Input import FileInput
from GUI.DataHandlerWrapper import *
from GUI.fileHandling import *
class Tab(QWidget):
    def __init__(self, name, column_1: List[Module], column_2: List[Module], column_2_display=True):
        super().__init__()
        self.name = name
        self.column_1 = column_1
        self.column_2 = column_2
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
def PreprocessingTab(main_widget):
    process_button = QPushButton()
    process_button.importance =1
    process_button.setText("Apply Settings")
    thread = PreprocessThread(main_widget,process_button)
    main_widget.thread_list.append(thread)
    process_button.clicked.connect(lambda: thread.runThread())

    return Tab("Preprocessing", column_1=[preprocessing_settings(main_widget), process_button], column_2=[
            main_widget.preprocess_image_view])

def ROIExtractionTab(main_widget):
    def click(event):
        event.accept()
        pos = event.pos()
        print(int(pos.x()), int(pos.y()))
        print(main_widget.roi_image_view.image_view.getImageItem().mapFromScene(pos))
    process_button = QPushButton()
    process_button.importance = 1
    process_button.setText("Apply Settings")
    thread = ROIExtractionThread(main_widget, process_button)
    main_widget.thread_list.append(thread)
    process_button.clicked.connect(lambda: thread.runThread())
    main_widget.roi_image_view.image_view.getImageItem().mouseClickEvent = click
    tab_selector = QTabWidget()
    tab_selector.setMaximumWidth(400)
    tab_selector.setStyleSheet("QTabWidget {font-size: 20px;}")
    tab_selector.importance = 1
    tab_selector.addTab(Tab("ROI Settings", column_1=[roi_extraction_settings(main_widget), process_button], column_2=[],column_2_display=False)," ROI Settings")
    tab_selector.addTab(QWidget(),"ROI List")
    return Tab("ROI Extraction", column_1=[tab_selector],
                         column_2=[main_widget.roi_image_view])
def AnalysisTab(main_widget):
    return Tab("Analysis", column_1=[], column_2=[])
def FileOpenTab(main_widget):
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
    tab_selector = QTabWidget()
    tab_selector.importance = 1
    tab_selector.addTab(file_open, file_open.name)
    tab_selector.addTab(folder_open, folder_open.name)
    tab_selector.addTab(prev_session_open, prev_session_open.name)

    return Tab("FileOpenTab", column_1=[tab_selector], column_2=[], column_2_display=False)