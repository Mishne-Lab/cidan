import logging
import os

from CIDAN.GUI.Data_Interaction.DataHandler import DataHandler
from CIDAN.GUI.Inputs.FileInput import createFileDialog

logger1 = logging.getLogger("CIDAN.loadDataset")


# def loadImageWrapper(main_widget):
#     def loadImage():
#         path_to_file = createFileDialog("~/Desktop", forOpen=True, fmt='', isFolder=0)
#         path_to_save_dir = createFileDialog("~/Desktop", forOpen=False, fmt='',
#                                             isFolder=1)
#         if hasattr(main_widget, "data_handler"):
#             main_widget.data_handler.__del__()
#         main_widget.data_handler = DataHandler(data_path=path_to_file,
#                                                save_dir_path=path_to_save_dir,
#                                                save_dir_already_created=False)
#         main_widget.image_view.setImage(main_widget.data_handler.calculate_filters())
#
#     return loadImage


def load_new_dataset(main_widget, file_input, save_dir_input, trials=None):
    print(trials)
    file_path = file_input.current_state()
    save_dir_path = save_dir_input.current_state()
    # if hasattr(main_widget, "data_handler"):
    #     main_widget.data_handler.__del__()
    if not trials:
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
    if trials:
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
