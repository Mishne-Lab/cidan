import sys

from qtpy import QtCore
from qtpy.QtWidgets import *


class Stream(QtCore.QObject):
    newText = QtCore.Signal(str)

    def __init__(self):
        super().__init__()

        # print(type(self.newText))
        # self.connect(self.newText)

    def write(self, text):
        self.newText.emit(str(text))

    def flush(self):
        pass


class ConsoleWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.label_start = QLabel()
        self.label_start.setStyleSheet("font-size: 15pt")
        self.label = QLabel()
        self.label.setStyleSheet("font-size: 15pt")
        self.label.setText("")
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("font-size: 15pt")
        self.progress_bar.hide()
        self.label_start.hide()
        self.layout.addWidget(self.label_start, alignment=QtCore.Qt.AlignRight,
                              stretch=20)
        self.layout.addWidget(self.progress_bar, alignment=QtCore.Qt.AlignRight,
                              stretch=1)
        self.layout.addWidget(self.label, alignment=QtCore.Qt.AlignRight, stretch=1)

        # self.setMinimumHeight(100)
        # self.process = QTextBrowser()
        # self.process.moveCursor(QtGui.QTextCursor.Start)
        # self.process.ensureCursorVisible()
        # self.process.setLineWrapColumnOrWidth(500)
        # self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)
        #
        # self.layout.addWidget(self.process)
        self.setLayout(self.layout)
        # sys.stdout = Stream()
        # sys.stdout.newText.connect(self.onUpdateText)
        self.update = True

    def updateText(self, text, warning=False):
        # self.label.show()
        self.progress_bar.hide()
        self.label.show()
        self.label_start.hide()
        self.label.setText(text)
        if warning:
            self.label.setStyleSheet("font-size: 15pt;background-color:#CC2936;")
        else:
            self.label.setStyleSheet("font-size: 15pt;")
        # if self.update:
        #     self.update =False
        #     print(text)
        #     self.update = True

    def updateProgressBar(self, start_text, percent):
        self.label_start.show()
        self.label.hide()
        self.progress_bar.show()
        self.label_start.setText(start_text)
        self.progress_bar.setValue(percent)
    def __del__(self):
        sys.stdout = sys.__stdout__
