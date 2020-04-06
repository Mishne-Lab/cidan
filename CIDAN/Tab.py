from PySide2.QtWidgets import *
from typing import Union, Any, List, Optional, cast, Tuple, Dict
from CIDAN.SettingsModule import *
from CIDAN.roiTools import *
from CIDAN.ImageViewModule import ImageViewModule
from CIDAN.Input import FileInput
from CIDAN.DataHandlerWrapper import *
from CIDAN.fileHandling import *
import pyqtgraph as pg
from CIDAN.ROIListModule import *
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
