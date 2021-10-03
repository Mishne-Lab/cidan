from typing import List

from cidan.GUI.ListWidgets.ROIListModule import *


class Column(QWidget):  # just used for stylesheets
    pass


class Tab(QWidget):
    def __init__(self, name, column_1: List[QFrame], column_2: List[QFrame],
                 column_2_display=True, column2_moveable=False, horiz_moveable=False):
        super().__init__()
        self.name = name
        self.column_1 = column_1
        self.column_2 = column_2
        self.setMinimumHeight(500)
        if horiz_moveable:
            self.layout = QSplitter(Qt.Horizontal)
            self.layout.setStyleSheet(
                "QSplitter { background-color: #19232D; } ")
        else:
            self.layout = QHBoxLayout()  # Main layout class
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.column_1_widget = Column()

        self.column_1_layout = QVBoxLayout()  # Layout for column 1
        self.column_1_layout.setContentsMargins(0, 0, 0, 0)
        self.column_1_widget.setLayout(self.column_1_layout)
        self.column_1_widget.setStyleSheet("Column { border:1px solid rgb(50, 65, "
                                           "75);} ")
        for module in column_1:
            if issubclass(module.__class__, QWidget):
                self.column_1_layout.addWidget(module)
            else:
                self.column_1_layout.addLayout(module)
        if horiz_moveable:
            self.layout.addWidget(self.column_1_widget)

        else:
            self.layout.addWidget(self.column_1_widget, stretch=1)
        if column_2_display:
            if column2_moveable:
                self.column_2_slider = QSplitter(Qt.Vertical)
                self.column_2_slider.setStyleSheet(
                    "QSplitter { background-color: #19232D; } ")
                for module in column_2:
                    self.column_2_slider.addWidget(module)
                if horiz_moveable:
                    self.layout.addWidget(self.column_2_slider)
                else:
                    self.layout.addWidget(self.column_2_slider, stretch=15)
            else:
                self.column_2_layout = QVBoxLayout()  # Layout for column 2
                self.column_2_layout.setContentsMargins(0, 0, 0, 0)

                self.column_2_widget = Column()
                self.column_2_widget.setStyleSheet(
                    "Column { border:1px solid rgb(50, 65, "
                    "75);} ")
                self.column_2_widget.setLayout(self.column_2_layout)
                for module in column_2:
                    self.column_2_layout.addWidget(module)
                if horiz_moveable:
                    self.layout.addWidget(self.column_2_widget)
                else:
                    self.layout.addWidget(self.column_2_widget, stretch=15)
        if horiz_moveable:
            layout = QVBoxLayout()
            layout.addWidget(self.layout)
            self.setLayout(layout)
        else:
            self.setLayout(self.layout)
