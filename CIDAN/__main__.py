import logging
import sys

from PySide2.QtWidgets import QApplication

from CIDAN.GUI import MainWindow

LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}

if len(sys.argv) > 1:
    LOG_FILENAME = 'log.out'
    level_name = sys.argv[1]
    level = LEVELS.get(level_name, logging.NOTSET)
    logging.basicConfig(filename=LOG_FILENAME, level=level)
    logger = logging.getLogger("CIDAN")
    logger.debug("Program started")

app = QApplication([])
app.setApplicationName("CIDAN")
widget = MainWindow.MainWindow()

sys.exit(app.exec_())
