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

class BoolSignal(QObject):
    def __init__(self):
        super().__init__()

    sig = Signal(bool)
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
            print("Starting preprocessing sequence")
            self.button.setEnabled(False)
            self.start()
    def endThread(self,image_data):
        print("Finished preprocessing sequence")
        self.button.setEnabled(True)
        self.main_widget.preprocess_image_view.setImage(image_data)
class ROIExtractionThread(Thread):
    def __init__(self,main_widget, button, roi_list_module, roi_tab):
        super().__init__(main_widget.data_handler)
        self.signal = BoolSignal()
        self.roi_tab = roi_tab
        self.main_widget = main_widget
        self.roi_list_module = roi_list_module
        self.button = button
        self.signal.sig.connect(lambda x: self.endThread(x))
    def run(self):
        # TODO add  try except wrapper to this
        try:
            self.data_handler.calculate_roi_extraction()
            self.signal.sig.emit(True)
        except AssertionError as e:
            print(e)
            self.signal.sig.emit(False)

    def runThread(self):

        if not any([x.isRunning() for x in self.main_widget.thread_list]):
            print("Starting ROI extraction")
            self.button.setEnabled(False)
            self.start()
    def endThread(self, success):
        self.button.setEnabled(True)
        if success:
            print("Finished ROI extraction")
            self.roi_list_module.set_list_items(self.main_widget.data_handler.clusters)
            self.roi_tab.current_image_flat = self.main_widget.data_handler.pixel_with_rois_color_flat
            self.roi_tab.updateImageDisplay()
