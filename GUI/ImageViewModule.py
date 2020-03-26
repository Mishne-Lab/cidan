from pyqtgraph import ImageView
from GUI.Module import Module
from PySide2.QtWidgets import *
from PySide2.QtCore import *
class ImageViewModule(Module):
    def __init__(self, main_widget):
        super().__init__(5)

        self.main_window = main_widget
        # self.setMinimumWidth(600)
        # self.setMinimumHeight(300)
        self.setStyleSheet("ImageViewModule {margin:5px; border:1px solid rgb(50, 65, "
                           "75);} ")
        self.layout = QHBoxLayout()
        # self.layout.setAlignment(Qt.AlignHCenter)

        self.setLayout(self.layout)
        # self.already_loaded = True
        # self.no_image_message = QPushButton("Please open a dataset first")
        # self.no_image_message.clicked.connect(main_widget.open_file_dialog)
        # self.no_image_message.setStyleSheet("QPushButton {font-size:80;}")
        self.image_view = ImageView()
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

