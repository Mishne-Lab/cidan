from PySide2.QtWidgets import *
from pyqtgraph import ImageView


class ImageViewModule(QFrame):
    def __init__(self, main_widget, histogram=True, roi=True):
        super().__init__()

        self.main_window = main_widget
        # self.setMinimumWidth(600)
        # self.setMinimumHeight(300)
        # self.setStyleSheet("ImageViewModule {margin:5px; border:1px solid rgb(50, 65, "
        #                    "75);} ")
        self.setStyleSheet("ImageViewModule {margin:0px; border:0px  solid rgb(50, 65, "
                           "75); padding: 0px;} ")
        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)

        # self.layout.setAlignment(Qt.AlignHCenter)

        self.setLayout(self.layout)
        # self.already_loaded = True
        # self.no_image_message = QPushButton("Please open a dataset first")
        # self.no_image_message.clicked.connect(main_widget.open_file_dialog)
        # self.no_image_message.setStyleSheet("QPushButton {font-size:80;}")
        self.image_view = ImageView()
        self.image_view.ui.layoutWidget.setContentsMargins(0, 0, 0, 0)
        # self.image_view.ui.roiBtn.hide()
        self.image_view.ui.menuBtn.hide()
        if not histogram:
            self.image_view.ui.histogram.hide()
        if not roi:
            self.image_view.ui.roiBtn.hide()
        # self.image_view.getRoiPlot().hide()

        self.layout.addWidget(self.image_view)

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
