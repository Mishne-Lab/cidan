import argparse
import logging
import sys

from qtpy.QtWidgets import QApplication

from CIDAN.GUI import MainWindow
from CIDAN.GUI.Data_Interaction import DataHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev", type=bool, default=False, required=False,
                        help="Enable Developement mode")
    LOG_FILENAME = 'log.out'
    parser.add_argument("-lp", "--logpath", type=str, default=LOG_FILENAME,
                        required=False, help="Path to save log file")
    parser.add_argument("--headless", type=bool, default=False,
                        required=False, help="Whether to run software headless")
    parser.add_argument("-p", "--parameter", type=str, default=False,
                        required=False, help="Path to parameter file")
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
    if args.headless:
        data_handler = DataHandler.DataHandler("",
                                               "",
                                               trials=[],
                                               save_dir_already_created=True,
                                               parameter_file=args.parameter)
        data_handler.calculate_filters()
        data_handler.calculate_roi_extraction()
        data_handler.export()
    else:
        app = QApplication([])
        app.setApplicationName("CIDAN")
        widget = MainWindow.MainWindow(dev=args.dev)

        sys.exit(app.exec_())


if __name__ == '__main__':
    main()
