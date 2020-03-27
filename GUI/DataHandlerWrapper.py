from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
import numpy as np

class MatrixSignal(QObject):
    def __init__(self):
        super().__init__()
    sig = Signal(np.matrix)
class StrSignal(QObject):
    def __init__(self):
        super().__init__()
    sig = Signal(str)
class Thread(QThread):
    def __init__(self, data_handler, parent=None):
        QThread.__init__(self, parent)
        self.exiting = False
        self.signal = MatrixSignal()
        self.data_handler = data_handler

    def run(self):
        pass

class PreprocessThread(Thread):
    def __init__(self, main_widget, button):
        super().__init__(main_widget.data_handler)
        self.main_widget = main_widget
        self.button = button
        self.signal.sig.connect(lambda x: self.endThread(x))

    def run(self):
        self.signal.sig.emit(self.data_handler.calculate_filters())
    def runThread(self):
        if not any([x.isRunning() for x in self.main_widget.thread_list]):
            self.button.setEnabled(False)
            self.start()
    def endThread(self,image_data):
        self.button.setEnabled(True)
        self.main_widget.image_view.setImage(image_data)
class ROIExtractionThread(Thread):
    def __init__(self,data_handler):
        super().__init__(data_handler)
        self.signal = StrSignal()
    def run(self):
        self.data_handler.calculate_filters()
        self.signal.sig.emit("")