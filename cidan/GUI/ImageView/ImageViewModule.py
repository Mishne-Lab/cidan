import numpy as np
from pyqtgraph import ImageView, PlotItem
from qtpy import QtCore
from qtpy.QtWidgets import *


class ImageViewModule(QFrame):
    """
    This class wraps the pyqt imageview model, takes care of configuring it and adds
    a set image method to it
    """

    def __init__(self, main_widget, histogram=True, crop_selector=False):
        super().__init__()

        self.main_widget = main_widget
        # self.setMinimumWidth(600)
        # self.setMinimumHeight(300)
        # self.setStyleSheet("ImageViewModule {margin:5px; border:1px solid rgb(50, 65, "
        #                    "75);} ")
        self.setStyleSheet("ImageViewModule {margin:0px; border:0px  solid rgb(50, 65, "
                           "75); padding: 0px;} ")
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        # self.layout.setAlignment(Qt.AlignHCenter)
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)
        # self.already_loaded = True
        # self.no_image_message = QPushButton("Please open a dataset first")
        # self.no_image_message.clicked.connect(main_widget.open_file_dialog)
        # self.no_image_message.setStyleSheet("QPushButton {font-size:80;}")
        self.plot_item = PlotItem()
        self.image_view = ImageView(view=self.plot_item)
        self.image_view.keyPressEvent = self.keyPressEvent
        self.image_view.ui.layoutWidget.setContentsMargins(0, 0, 0, 0)
        # self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        if not histogram:
            self.image_view.ui.histogram.hide()
        if not crop_selector:
            self.image_view.ui.roiBtn.hide()
        # self.image_view.getRoiPlot().hide()
        self.image_item = self.image_view.getImageItem()

        self.layout.addWidget(self.image_view)

    @property
    def data_handler(self):
        return self.main_widget.data_handler
    def keyPressEvent(self, ev):
        if ev.key() == QtCore.Qt.Key_Space and False:
            if self.image_view.playRate == 0:
                fps = (self.image_view.getProcessedImage().shape[0] - 1) / (
                            self.image_view.tVals[-1] - self.image_view.tVals[0])
                self.image_view.play(fps)
                # print fps
            else:
                self.image_view.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_Home:
            self.image_view.setCurrentIndex(0)
            self.image_view.play(0)
            ev.accept()
        elif ev.key() == QtCore.Qt.Key_End:
            self.image_view.setCurrentIndex(
                self.image_view.getProcessedImage().shape[0] - 1)
            self.image_view.play(0)
            ev.accept()
        elif ev.key() in self.image_view.noRepeatKeys:
            ev.accept()
            if ev.isAutoRepeat():
                return
            self.image_view.keysPressed[ev.key()] = 1
            self.image_view.evalKeyState()
        else:
            QWidget.keyPressEvent(self.image_view, ev)
    def setImage(self, data):

        # if self.already_loaded == False:
        #     print("changed image")
        #     self.already_loaded = True
        #     self.layout.removeWidget(self.no_image_message)
        #     self.no_image_message.deleteLater()
        #     # self.layout.setAlignment(Qt.AlignLeft)
        #     self.image_view = ImageView()
        #
        #     self.layout.addWidget(self.image_view)
        self.image_view.setImage(data, levelMode='mono', autoRange=True,
                                 autoLevels=True, autoHistogramRange=True)
        bottom_5 = np.percentile(data, 5)
        top_5 = np.percentile(data, 95)
        bottom_2 = np.percentile(data, 2)
        top_2 = np.percentile(data, 98)
        self.image_view.setLevels(bottom_5, top_5)
        self.image_view.setHistogramRange(bottom_2, top_2)
