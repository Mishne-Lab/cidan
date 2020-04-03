from PySide2.QtWidgets import QMainWindow, QWidget, QApplication, QVBoxLayout, \
    QHBoxLayout, QPushButton, QTabWidget
from GUI.Tab import Tab,  AnalysisTab, FileOpenTab
from GUI.ROIExtractionTab import *
from GUI.PreprocessingTab import *
import qdarkstyle
from GUI.ImageViewModule import ImageViewModule
from GUI.fileHandling import loadImageWrapper
from GUI.DataHandler import DataHandler
from GUI.DataHandlerWrapper import Thread
from GUI.ConsoleWidget import ConsoleWidget
import sys


class MainWindow(QMainWindow):
    """Initializes the basic window with Main widget being the focused widget"""

    def __init__(self):
        super().__init__()
        self.title = 'Mishne Lab-ROI Extraction Suite'
        self.width = 900
        self.height = 400
        self.setWindowTitle(self.title)
        self.setMinimumSize(self.width, self.height)
        self.table_widget = MainWidget(self)
        self.setCentralWidget(self.table_widget)
        # self.setStyleSheet(qdarkstyle.load_stylesheet())
        with open("main_stylesheet.css", "r") as f:
            self.setStyleSheet(qdarkstyle.load_stylesheet() + f.read())
        self.show()


class MainWidget(QWidget):
    """Main Widget, contains everything

    Attributes
    ----------
    layout : QLayout
        The main layout for the widget
    data_handler : DataHandler
        The instance that controls all interactions with dataset
    thread_list : List[Thread]
        A list of all the possible running threads, used to ensure only 1 thread is
        running at a time
    preprocess_image_view : ImageViewModule
        The image view for the preprocess tab
    roi_image_view : ImageViewModule
        The image view for the roi extraction tab
    tab_widget : QTabWidget
        Controls the main tabs of the application
    console : ConsoleWidget
        Widget for the console
    tabs : List[Tabs]
        A list of the currently active tabs not used until after init_w_data is run
    """

    def __init__(self, parent):
        """
        Initialize the main widget to load files
        Parameters
        ----------
        parent
        """
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.data_handler = None
        self.thread_list = []
        self.preprocess_image_view = ImageViewModule(self)
        self.roi_image_view = ImageViewModule(self, histogram=False)

        self.tab_widget = QTabWidget()
        self.tab_widget.addTab(FileOpenTab(self), "Open Dataset")
        # This part add placeholder tabs until data is loaded
        self.tabs = ["Preprocessing", "ROI Extraction", "Analysis"]
        for num, tab in enumerate(self.tabs):
            self.tab_widget.addTab(QWidget(), tab)
            self.tab_widget.setTabEnabled(num + 1, False)
        self.layout.addWidget(self.tab_widget)

        self.console = ConsoleWidget()
        self.layout.addWidget(self.console)
        self.setLayout(self.layout)

        # Below here in this function is just code for testing
        # TODO check if it can load data twice
        if True:
            # auto loads a small dataset
            self.data_handler = DataHandler(
                "/Users/sschickler/Documents/LSSC-python/input_images/small_dataset.tif",
                "/Users/sschickler/Documents/LSSC-python/input_images/test31",
                save_dir_already_created=True)
            self.init_w_data()
        if False:
            # auto loads a large dataset
            self.data_handler = DataHandler(
                "/Users/sschickler/Documents/LSSC-python/input_images/dataset_1",
                "/Users/sschickler/Documents/LSSC-python/input_images/test3",
                save_dir_already_created=False)
            self.init_w_data()

    def init_w_data(self):
        """
        Initialize main widget with data

        It activates the other tabs and helps load the data into image views
        Returns
        -------

        """
        for num, _ in enumerate(self.tabs):
            self.tab_widget.removeTab(1)
        # TODO actually delete the tabs not just remove them

        self.tabs = [PreprocessingTab(self), ROIExtractionTab(self), AnalysisTab(self)]

        # Add tabs
        for tab in self.tabs:
            self.tab_widget.addTab(tab, tab.name)
        self.tab_widget.setCurrentIndex(1)


if __name__ == "__main__":
    app = QApplication([])

    widget = MainWindow()

    sys.exit(app.exec_())
