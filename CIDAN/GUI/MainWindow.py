import os

from CIDAN.GUI.Tabs.AnalysisTab import AnalysisTab

os.environ['QT_API'] = 'pyside2'
from qtpy.QtWidgets import QTabWidget
from CIDAN.GUI.Tabs.FileOpenTab import FileOpenTab
from CIDAN.GUI.Tabs.ROIExtractionTab import *
from CIDAN.GUI.Tabs.PreprocessingTab import *
import qdarkstyle
from CIDAN.GUI.ImageView.ImageViewModule import ImageViewModule
from CIDAN.GUI.Data_Interaction.DataHandler import DataHandler
from CIDAN.GUI.Console.ConsoleWidget import ConsoleWidget
import sys
import logging
import pyqtgraph as pg

# sys._excepthook = sys.excepthook
# def exception_hook(exctype, value, traceback):
#     print(exctype, value, traceback)
#     sys._excepthook(exctype, value, traceback)
#     sys.exit(1)
# sys.excepthook = exception_hook

pg.setConfigOption("imageAxisOrder", "row-major")

class MainWindow(QMainWindow):
    """Initializes the basic window with Main widget being the focused widget"""

    def __init__(self, dev=False, preload=False):
        super().__init__()
        self.title = 'CIDAN'
        self.width = 900
        self.height = 800
        self.setWindowTitle(self.title)
        self.setMinimumSize(self.width, self.height)
        self.main_menu = self.menuBar()
        self.table_widget = MainWidget(self, dev=dev, preload=preload)
        self.setCentralWidget(self.table_widget)
        # self.setStyleSheet(qdarkstyle.load_stylesheet())
        style = """
            QTabWidget {font-size: 25px; padding:1px; margin:5px;}
            QTabBar::tab {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                           stop: 0 #E1E1E1, stop: 0.4 #DDDDDD,
                                          stop: 0.5 #D8D8D8, stop: 1.0 #D3D3D3);
                /*border: 2px solid #C4C4C3;*/
                /*border-bottom-color: #C2C7CB; !* same as the pane color *!*/
                
                min-width: 8ex;
                padding:1px;
                border:1px;
            }
            
            QComboBox::item:checked {
              font-weight: bold;
              height: 12px;
            }
            """
        self.setStyleSheet(qdarkstyle.load_stylesheet() + style)

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

    def __init__(self, parent, dev=False, preload=True):
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

        self.dev = dev
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
        self.console.setMaximumHeight(150)
        self.console.setMinimumHeight(150)
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
        if preload and dev:
            try:
                # auto loads a small dataset
                self.data_handler = DataHandler(

                    "/Users/sschickler/Code_Devel/LSSC-python/input_images/",
                    "/Users/sschickler/Documents/LSSC-python",
                    trials=["small_dataset1.tif"],
                    save_dir_already_created=True)
                self.init_w_data()
            except AttributeError:
                pass
        if False and dev:
            # auto loads a large dataset
            self.data_handler = DataHandler(
                "/Users/sschickler/Code Devel/LSSC-python/input_images/dataset_1",
                "/Users/sschickler/Code Devel/LSSC-python/input_images/test3",
                save_dir_already_created=False)
            self.init_w_data()

    def init_w_data(self):
        """
        Initialize main widget with data

        It activates the other tabs and helps load the data into image views
        Returns
        -------

        """
        self.thread_list = []
        self.preprocess_image_view = ImageViewModule(self)

        for num, _ in enumerate(self.tabs):
            self.tab_widget.removeTab(1)

        # TODO add to export tab to export all time traces or just currently caclulated ones
        self.tabs = [PreprocessingTab(self), ROIExtractionTab(self), AnalysisTab(self)]

        # Add tabs
        for tab in self.tabs:
            self.tab_widget.addTab(tab, tab.name)
        self.tab_widget.setCurrentIndex(1)
        self.image_view_list = [x.image_view for x in self.tabs]
        # self.tab_widget.currentChanged.connect(
        #     lambda x: self.tabs[1].image_view.set_background("",
        #                                                      self.tabs[
        #                                                          1].image_view.background_chooser.current_state(),
        #                                                      update_image=True))
        # self.tab_widget.currentChanged.connect(
        #     lambda x: self.tabs[2].image_view.reset_view())
        # self.tab_widget.currentChanged.connect(
        #     lambda x: self.tabs[2].reset_view())
        if not hasattr(self, "export_menu"):
            self.export_menu = self.main_menu.addMenu("&Export")
            export_action = QAction("Export Time Traces/ROIs", self)
            export_action.setStatusTip('Export Time Traces/ROIs')
            export_action.triggered.connect(lambda: self.exportStuff())
            self.export_menu.addAction(export_action)

    def selectOpenFileTab(self, index):
        self.tab_widget.setCurrentIndex(0)
        self.fileOpenTab.tab_selector.setCurrentIndex(index)

    def exportStuff(self):
        dialog = QDialog()
        dialog.setStyleSheet(qdarkstyle.load_stylesheet())
        dialog.layout = QVBoxLayout()
        dialog.setWindowTitle("Select Trials to Export")
        trial_dialog = TrialListWidget()
        trial_dialog.set_items_from_list(self.data_handler.trials_all,
                                         trials_selected_indices=self.data_handler.trials_loaded_time_trace_indices)

        def export_func():
            self.data_handler.update_selected_trials(
                trial_dialog.selectedTrials())
            self.data_handler.export()
            dialog.close()
            msg = QMessageBox()
            msg.setStyleSheet(qdarkstyle.load_stylesheet())
            msg.setWindowTitle("Export data")

            msg.setText("Data Exported to save directory: "+  self.data_handler.save_dir_path)
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()

        dialog.layout.addWidget(trial_dialog)
        export_button = QPushButton("Export")
        export_button.clicked.connect(lambda x: export_func())

        dialog.layout.addWidget(export_button)
        dialog.setLayout(dialog.layout)
        dialog.show()

    def checkThreadRunning(self):
        if (any([x.isRunning() for x in self.thread_list])):
            msg = QMessageBox()
            msg.setStyleSheet(qdarkstyle.load_stylesheet())
            msg.setWindowTitle("Operation Denied")

            msg.setText(
                "Sorry we can't preform this operation until current process is "
                "finished")
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()
            return False

        else:
            return True

    def updateTabs(self):
        for x in self.tabs:
            x.updateTab()


if __name__ == "__main__":
    # client = Client(processes=False, threads_per_worker=16,
    #                 n_workers=1, memory_limit='32GB')
    # print(client)
    LOG_FILENAME = 'log.out'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)
    logger = logging.getLogger("CIDAN")
    logger.debug("Program started")
    app = QApplication([])
    app.setApplicationName("CIDAN")
    widget = MainWindow(dev=True, preload=True)

    sys.exit(app.exec_())
