from CIDAN.DataHandler import DataHandler
from PySide2.QtWidgets import *
# from PySide2.QtGui import *
from PySide2 import QtCore
import logging
logger1 = logging.getLogger("CIDAN.fileHandling")
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

def createFileDialog(directory='', forOpen=True, fmt='', isFolder=0):
    directory="/Users/sschickler/Documents/LSSC-python/input_images"
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    options |= QFileDialog.DontUseCustomDirectoryIcons
    dialog = QFileDialog()
    dialog.setOptions(options)

    dialog.setFilter(dialog.filter() | QtCore.QDir.Hidden)

    # ARE WE TALKING ABOUT FILES OR FOLDERS
    if isFolder == 1:
        dialog.setFileMode(QFileDialog.DirectoryOnly)
    if isFolder == 2:
        dialog.setFileMode(QFileDialog.AnyFile)
    # OPENING OR SAVING
    dialog.setAcceptMode(QFileDialog.AcceptOpen) if forOpen else dialog.setAcceptMode(QFileDialog.AcceptSave)

    # SET FORMAT, IF SPECIFIED
    if fmt != '' and isFolder is False:
        dialog.setDefaultSuffix(fmt)
        dialog.setNameFilters([f'{fmt} (*.{fmt})'])

    # SET THE STARTING DIRECTORY
    if directory != '':
        dialog.setDirectory(str(directory))
    # else:
    #     dialog.setDirectory(str(ROOT_DIR))


    if dialog.exec_() == QDialog.Accepted:
        path = dialog.selectedFiles()[0]  # returns a list
        return path
    else:
        return ''

def load_new_dataset(main_widget, file_input, save_dir_input,trials=None):
    print(trials)
    file_path = file_input.current_state()
    save_dir_path = save_dir_input.current_state()
    # if hasattr(main_widget, "data_handler"):
    #     main_widget.data_handler.__del__()
    if not trials:
        try:

            main_widget.data_handler = DataHandler(data_path="",trials=[file_path],
                                                   save_dir_path=save_dir_path,
                                               save_dir_already_created=False)
            main_widget.init_w_data()
        except Exception as e:
            logger1.error(e)
            print("Loading Failed please make sure it is a valid file")
    if trials:
        logger1.debug("Trials:"+str(trials))
        if len(trials) == 0:
            print("Please select at least one trial")
        try:
            main_widget.data_handler = DataHandler(data_path=file_path, trials=trials,
                                               save_dir_path=save_dir_path,
                                               save_dir_already_created=False)
            main_widget.init_w_data()
        except Exception as e:
            logger1.error(e)
            print("Loading Failed please make sure it is a valid folder and all trials"
                  + "are valid files")


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
    createFileDialog(forOpen=False,isFolder=2)
