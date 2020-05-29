import argparse
import logging
import sys

from qtpy.QtWidgets import QApplication

from CIDAN.GUI import MainWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev", type=bool, default=False, required=False,
                        help="Enable Developement mode")
    LOG_FILENAME = 'log.out'
    parser.add_argument("-lp", "--logpath", type=str, default=LOG_FILENAME,
                        required=False, help="Path to save log file")
    parser.add_argument("-ll", "--loglevel", type=str, default="error",
                        required=False,
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}

    level = LEVELS.get(args.loglevel, logging.NOTSET)
    logging.basicConfig(filename=args.logpath, level=level)
    logger = logging.getLogger("CIDAN")
    logger.debug("Program started")
    dev = False
    if len(sys.argv) > 1:
        dev = True if sys.argv[1] == "True" else False
    print(dev)
    app = QApplication([])
    app.setApplicationName("CIDAN")
    widget = MainWindow.MainWindow(dev=dev)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
