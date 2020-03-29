from PySide2.QtWidgets import QMainWindow, QWidget, QApplication, QVBoxLayout, \
    QHBoxLayout, QPushButton, QTabWidget
from GUI.Tab import Tab, PreprocessingTab, AnalysisTab, ROIExtractionTab, FileOpenTab
import qdarkstyle
from GUI.ImageViewModule import ImageViewModule
from GUI.fileHandling import loadImageWrapper
from GUI.DataHandler import DataHandler
from GUI.ConsoleWidget import ConsoleWidget
import sys
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Mishne Lab-ROI Extraction Suite'
        self.width = 900
        self.height = 400
        self.setWindowTitle(self.title)
        self.setMinimumSize(self.width, self.height)
        self.table_widget = MainWidget(self)
        self.setCentralWidget(self.table_widget)
        self.setStyleSheet(qdarkstyle.load_stylesheet())
        with open("main_stylesheet.css", "r") as f:
            self.setStyleSheet(qdarkstyle.load_stylesheet()+f.read())
        self.show()
class MainWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.data_handler = None
        self.thread_list = []
        self.open_file_dialog = loadImageWrapper(self)
        self.preprocess_image_view = ImageViewModule(self)
        self.roi_image_view = ImageViewModule(self, histogram=False)
        self.tab_widget = QTabWidget()
        self.tabs = ["Preprocessing", "ROI Extraction", "Analysis"]
        # Add tabs
        self.tab_widget.addTab(FileOpenTab(self), "Open Dataset")
        for num, tab in enumerate(self.tabs):
            self.tab_widget.addTab(QWidget(),tab)
            self.tab_widget.setTabEnabled(num+1, False)

        self.layout.addWidget(self.tab_widget)
        self.console = ConsoleWidget()
        self.layout.addWidget(self.console)
        self.setLayout(self.layout)
        # TODO check if it can load data twice
        if True:
            self.data_handler = DataHandler("/Users/sschickler/Documents/LSSC-python/input_images/small_dataset.tif","/Users/sschickler/Documents/LSSC-python/input_images/test31", save_dir_already_created=True)
            self.init_w_data()
        if False:
            self.data_handler = DataHandler("/Users/sschickler/Documents/LSSC-python/input_images/dataset_1","/Users/sschickler/Documents/LSSC-python/input_images/test3", save_dir_already_created=False)
            self.init_w_data()
    def init_w_data(self):
        #
        #
        #
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