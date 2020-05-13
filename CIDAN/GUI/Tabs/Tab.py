from typing import List

from CIDAN.GUI.ListWidgets.ROIListModule import *


class Column(QWidget):  # just used for stylesheets
    pass


class Tab(QWidget):
    def __init__(self, name, column_1: List[QFrame], column_2: List[QFrame],
                 column_2_display=True):
        super().__init__()
        self.name = name
        self.column_1 = column_1
        self.column_2 = column_2
        self.setMinimumHeight(500)
        self.layout = QHBoxLayout()  # Main layout class
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.column_1_widget = Column()

        self.column_1_layout = QVBoxLayout()  # Layout for column 1
        self.column_1_layout.setContentsMargins(0, 0, 0, 0)
        self.column_1_widget.setLayout(self.column_1_layout)
        self.column_1_widget.setStyleSheet("Column { border:1px solid rgb(50, 65, "
                                           "75);} ")
        for module in column_1:
            self.column_1_layout.addWidget(module)
        self.layout.addWidget(self.column_1_widget, stretch=1)
        if column_2_display:
            self.column_2_layout = QVBoxLayout()  # Layout for column 2
            self.column_2_layout.setContentsMargins(0, 0, 0, 0)

            self.column_2_widget = Column()
            self.column_2_widget.setStyleSheet("Column { border:1px solid rgb(50, 65, "
                                               "75);} ")
            self.column_2_widget.setLayout(self.column_2_layout)
            for module in column_2:
                self.column_2_layout.addWidget(module)
            self.layout.addWidget(self.column_2_widget, stretch=2)
        self.setLayout(self.layout)


class AnalysisTab(Tab):
    def __init__(self, main_widget):
        super().__init__("Analysis", column_1=[], column_2=[])
