import logging
import os

from CIDAN.GUI.Data_Interaction.DataHandler import DataHandler
from CIDAN.GUI.Inputs.FileInput import createFileDialog

logger1 = logging.getLogger("CIDAN.loadDataset")


def load_new_dataset(main_widget, file_input, save_dir_input, trials=None,
                     single=False):
    """
    
    Parameters
    ----------
    main_widget
    file_input
    save_dir_input
    trials
    single

    Returns
    -------

    """

    file_path = file_input.current_state()
    save_dir_path = save_dir_input.current_state()

    if single:
        try:
            dir_path = os.path.dirname(file_path)

            main_widget.data_handler = DataHandler(data_path=dir_path,
                                                   trials=[os.path.basename(file_path)],
                                                   save_dir_path=save_dir_path,
                                                   save_dir_already_created=False)
            main_widget.init_w_data()
        except IndentationError as e:
            logger1.error(e)
            print("Loading Failed please make sure it is a valid file")
    elif not trials:
        try:
            dir_path = os.path.dirname(file_path)

            main_widget.data_handler = DataHandler(data_path=dir_path,
                                                   trials=[os.path.basename(file_path)],
                                                   save_dir_path=save_dir_path,
                                                   save_dir_already_created=False)
            main_widget.init_w_data()
        except IndentationError as e:
            logger1.error(e)
            print("Loading Failed please make sure it is a valid file")
    elif trials:
        logger1.debug("Trials:" + str(trials))
        if len(trials) == 0:
            print("Please select at least one trial")
        try:
            main_widget.data_handler = DataHandler(data_path=file_path, trials=trials,
                                                   save_dir_path=save_dir_path,
                                                   save_dir_already_created=False)
            main_widget.init_w_data()
        except IndentationError as e:
            logger1.error(e)
            print("Loading Failed please make sure it is a valid folder and all trials"
                  + " are valid files")


def load_prev_session(main_widget, save_dir_input):
    # TODO add error handling here to ensure valid inputs
    save_dir_path = save_dir_input.current_state()
    try:
        main_widget.data_handler = DataHandler(data_path="",
                                               save_dir_path=save_dir_path,
                                               save_dir_already_created=True)
        main_widget.init_w_data()
    except Exception as e:
        logger1.error(e)
        print("Loading Failed please try again, if problem persists save directory " +
              "is corrupted")


def export_timetraces(main_widget):
    createFileDialog(forOpen=False, isFolder=2)
