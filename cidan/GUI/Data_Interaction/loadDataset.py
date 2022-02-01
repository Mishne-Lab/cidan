import logging
import os

import qdarkstyle
from qtpy import QtWidgets
from qtpy.QtWidgets import QMessageBox

from cidan.GUI.Inputs.FileInput import createFileDialog

logger1 = logging.getLogger("cidan.loadDataset")


def load_new_dataset(main_widget, file_path, save_dir_path, trials=None,
                     single=False, load_into_mem=True, override_load_warning=False,
                     mask_path=None):
    """
    Function to load a initialize a new DataHandler object and the GUI with the data
    Parameters
    ----------
    main_widget : MainWidget
        The main widget for the GUI
    file_path : str
        Either the folder that contains the trials or the file to load
    save_dir_path : str
        The path to the save directory
    trials : List[str]
        A list of trials either paths to files or folders
    single
        Whether to load the list of trials as a single one, ie like they are each only
        one time point so load the entire folder as one.
    override_many_files : bool
        Whether to give user dialog to choose whether to rearange files if there are over 250

    Returns
    -------
    Nothing
    """
    if (main_widget.checkThreadRunning()):

        if main_widget.data_handler is not None:
            main_widget.data_handler.__del__()
            main_widget.data_handler = None
            pass
        if save_dir_path == "":
            msg = QMessageBox()
            msg.setStyleSheet(qdarkstyle.load_stylesheet())
            msg.setWindowTitle("Error")

            msg.setText(
                "Please select a save directory for cidan to save its outputs in")
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()

            return
        if file_path == "":
            msg = QMessageBox()
            msg.setStyleSheet(qdarkstyle.load_stylesheet())
            msg.setWindowTitle("Error")

            msg.setText(
                "Please select a file for cidan")
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()

            return
        if mask_path == "" and main_widget.widefield:
            msg = QMessageBox()
            msg.setStyleSheet(qdarkstyle.load_stylesheet())
            msg.setWindowTitle("Error")

            msg.setText(
                "Please select a mask for cidan")
            msg.setIcon(QMessageBox.Information)
            x = msg.exec_()

            return
        if single:
            try:
                dir_path = os.path.dirname(file_path)
                main_widget.open_dataset_thread.runThread(data_path=dir_path,
                                                          trials=[
                                                              os.path.basename(
                                                                  file_path)],
                                                          save_dir_path=save_dir_path,
                                                          save_dir_already_created=False,
                                                          load_into_mem=load_into_mem,
                                                          mask_path=mask_path,
                                                          widefield=main_widget.widefield)

            except Exception as e:
                logger1.error(e)
                error_dialog = QtWidgets.QErrorMessage(main_widget.main_window)
                error_dialog.showMessage(
                    "Loading Failed please make sure it is a valid file")
                # error_dialog.exec_()

        elif not trials:
            try:
                dir_path = os.path.dirname(file_path)

                main_widget.open_dataset_thread.runThread(data_path=dir_path,
                                                          trials=[
                                                              os.path.basename(
                                                                  file_path)],
                                                          save_dir_path=save_dir_path,
                                                          save_dir_already_created=False,
                                                          load_into_mem=load_into_mem,
                                                          mask_path=mask_path,
                                                          widefield=main_widget.widefield)
            except Exception as e:
                logger1.error(e)
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage(
                    "Loading Failed please make sure it is a valid file")

        elif trials:
            logger1.debug("Trials:" + str(trials))
            if len(trials) == 0:
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage("Please select at least one trial")
                return
            try:
                main_widget.open_dataset_thread.runThread(data_path=file_path,
                                                          trials=trials,
                                                          save_dir_path=save_dir_path,
                                                          save_dir_already_created=False,
                                                          load_into_mem=load_into_mem,
                                                          mask_path=mask_path,
                                                          widefield=main_widget.widefield)
            except Exception as e:
                logger1.error(e)
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage(

                    "Loading Failed please make sure it is a valid folder and all trials"
                    + " are valid files")


def load_prev_session(main_widget, save_dir_path):
    """
    Loads a previous session into Datahandler and initializes GUI
    Parameters
    ----------
    main_widget : MainWidget
        The main widget of the application
    save_dir_path : str
        Path of the save directory

    Returns
    -------
    Nothing
    """
    if (main_widget.checkThreadRunning()):
        try:
            main_widget.open_dataset_thread.runThread(data_path="",
                                                      save_dir_path=save_dir_path,
                                                      save_dir_already_created=True,
                                                      load_into_mem=None, trials=None)


        except Exception as e:
            logger1.error(e)
            print(
                "Loading Failed please try again, if problem persists save directory " +
                "is corrupted")


def export_timetraces(main_widget):
    createFileDialog(forOpen=False, isFolder=2)
