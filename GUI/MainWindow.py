from PySide2.QtWidgets import QMainWindow, QWidget, QApplication, QVBoxLayout, \
    QHBoxLayout, QPushButton, QTabWidget
from GUI.Tab import Tab,  AnalysisTab
from GUI.FileOpenTab import FileOpenTab
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
        self.title = 'CIDAN'
        self.width = 900
        self.height = 400
        self.setWindowTitle(self.title)
        self.setMinimumSize(self.width, self.height)
        self.main_menu = self.menuBar()
        self.table_widget = MainWidget(self)
        self.setCentralWidget(self.table_widget)
        # self.setStyleSheet(qdarkstyle.load_stylesheet())
        with open("main_stylesheet.css", "r") as f:
            self.setStyleSheet(qdarkstyle.load_stylesheet() + f.read())

        # extractAction.triggered.connect()


        self.show()



class MainWidget(QWidget):
    """Main Widget, contains everything

    Attributes
    ----------
    main_window : MainWindow
        A reference to the main window of the application
    main_menu : ???
        the top bar menu
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
        self.main_window = parent
        self.main_menu = self.main_window.main_menu
        self.layout = QVBoxLayout(self)
        self.data_handler = None
        self.thread_list = []
        self.preprocess_image_view = ImageViewModule(self, roi=False)
        self.roi_image_view = ImageViewModule(self, histogram=False, roi=False)

        self.tab_widget = QTabWidget()
        self.fileOpenTab = FileOpenTab(self)
        self.tab_widget.addTab(self.fileOpenTab, "Open Dataset")

        # This part add placeholder tabs until data is loaded
        self.tabs = ["Preprocessing", "ROI Extraction", "Analysis"]
        for num, tab in enumerate(self.tabs):
            self.tab_widget.addTab(QWidget(), tab)
            self.tab_widget.setTabEnabled(num + 1, False)
        self.layout.addWidget(self.tab_widget)

        self.console = ConsoleWidget()
        self.layout.addWidget(self.console)
        self.setLayout(self.layout)

        # Initialize top bar menu
        fileMenu = self.main_menu.addMenu('&File')
        openFileAction = QAction("Open File", self)
        openFileAction.setStatusTip('Open a single file')
        openFileAction.triggered.connect(lambda: self.selectOpenFileTab(0))
        fileMenu.addAction(openFileAction)
        openFolderAction = QAction("Open Folder", self)
        openFolderAction.setStatusTip('Open a folder')
        openFolderAction.triggered.connect(lambda: self.selectOpenFileTab(1))
        fileMenu.addAction(openFolderAction)
        openPrevAction = QAction("Open Previous Session", self)
        openPrevAction.setStatusTip('Open a previous session')
        openPrevAction.triggered.connect(lambda: self.selectOpenFileTab(2))
        fileMenu.addAction(openPrevAction)



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
        self.export_menu = self.main_menu.addMenu("&Export")
        export_action = QAction("Export Time Traces/ROIs", self)
        export_action.setStatusTip('Export Time Traces/ROIs')
        export_action.triggered.connect(lambda: self.exportStuff())
        self.export_menu.addAction(export_action)
    def selectOpenFileTab(self, index):
        self.tab_widget.setCurrentIndex(0)
        self.fileOpenTab.tab_selector.setCurrentIndex(index)
    def exportStuff(self):
        msg = QMessageBox()
        msg.setWindowTitle("Export data")
        msg.setText("Data Exported to save directory")
        msg.setIcon(QMessageBox.Information)
        x = msg.exec_()
if __name__ == "__main__":
    app = QApplication([])
    app.setApplicationName("CIDAN")
    widget = MainWindow()

    sys.exit(app.exec_())
