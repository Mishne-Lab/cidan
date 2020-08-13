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
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel()
        self.label.setStyleSheet("font-size: 15pt")
        self.label.setText("")

        self.layout.addWidget(self.label, alignment=QtCore.Qt.AlignRight)
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

    def updateText(self, text):
        self.label.setText(text)
        # if self.update:
        #     self.update =False
        #     print(text)
        #     self.update = True

    def __del__(self):
        sys.stdout = sys.__stdout__
