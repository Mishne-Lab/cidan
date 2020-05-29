import sys

from qtpy import QtCore
from qtpy import QtGui
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
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.label = QLabel()
        self.label.setText("Console:")
        self.layout.addWidget(self.label)
        self.setMinimumHeight(100)
        self.process = QTextBrowser()
        self.process.moveCursor(QtGui.QTextCursor.Start)
        self.process.ensureCursorVisible()
        self.process.setLineWrapColumnOrWidth(500)
        self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)

        self.layout.addWidget(self.process)
        self.setLayout(self.layout)
        sys.stdout = Stream()
        sys.stdout.newText.connect(self.onUpdateText)

    def onUpdateText(self, text):
        cursor = self.process.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def __del__(self):
        sys.stdout = sys.__stdout__
