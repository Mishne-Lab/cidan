from CIDAN import MainWindow
from PySide2.QtWidgets import QApplication
import sys
app = QApplication([])
app.setApplicationName("CIDAN")
widget = MainWindow.MainWindow()

sys.exit(app.exec_())