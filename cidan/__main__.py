import argparse
import logging
import os
import sys

os.environ['QT_API'] = 'pyside2'
from qtpy.QtWidgets import QApplication


from cidan.GUI.Data_Interaction import DataHandler

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev", type=str2bool, default=False, required=False,
                        help="Enable Developement mode")
    LOG_FILENAME = 'log.out'
    parser.add_argument("-lp", "--logpath", type=str, default=LOG_FILENAME,
                        required=False, help="Path to save log file")
    parser.add_argument("--headless", type=str2bool, default=False,
                        required=False, help="Whether to run software headless")
    parser.add_argument("-p", "--parameter", type=str, default=False,
                        required=False, help="Path to parameter file")
    parser.add_argument("-ll", "--loglevel", type=str, default="error",
                        required=False,
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument("-w", "--widefield", type=str2bool, default=False,
                        required=False,
                        help="Enable widefield mode")
    args = parser.parse_args()
    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}

    level = LEVELS.get(args.loglevel, logging.NOTSET)
    logging.basicConfig(filename=args.logpath, level=level)
    logger = logging.getLogger("cidan")
    logger.debug("Program started")
    if args.headless:
        data_handler = DataHandler.DataHandler("",
                                               "",
                                               trials=[],
                                               save_dir_already_created=True,
                                               parameter_file=args.parameter,
                                               load_into_mem=True)
        data_handler.calculate_filters()
        data_handler.calculate_roi_extraction()
        data_handler.export()
    else:
        from cidan.GUI import MainWindow
        app = QApplication([])
        app.setApplicationName("cidan")
        widget = MainWindow.MainWindow(dev=args.dev, widefield=args.widefield)

        sys.exit(app.exec_())


if __name__ == '__main__':
    main()
