from qtpy.QtWidgets import *

from CIDAN.GUI.Inputs.Input import Input


class FileInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip, isFolder, forOpen):
        super().__init__(display_name, program_name, on_change_function,
                         default_val, tool_tip)
        self.isFolder = isFolder
        self.forOpen = forOpen
        self.path = ""
        self.current_location = QLabel()
        self.current_location.setText("")
        self.layout_h.addWidget(self.current_location)
        self.button = QPushButton()
        self.button.setText("Browse")
        self.button.clicked.connect(self.on_browse_button)
        self.button.setToolTip(tool_tip)
        self.layout_h.addWidget(self.button)

    def on_browse_button(self):
        self.path = createFileDialog(directory="~/Desktop", forOpen=self.forOpen,
                                     isFolder=self.isFolder)
        self.current_location.setText(self.path)
        if self.on_change_function is not None:
            self.on_change_function(self.path)

    def current_state(self):
        return self.path


def createFileDialog(directory='', forOpen=True, fmt='', isFolder=0):
    directory = "/Users/sschickler/Documents/LSSC-python/input_images"
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    options |= QFileDialog.DontUseCustomDirectoryIcons
    dialog = QFileDialog()
    dialog.setOptions(options)

    dialog.setFilter(dialog.filter())

    # ARE WE TALKING ABOUT FILES OR FOLDERS
    if isFolder == 1:
        dialog.setFileMode(QFileDialog.DirectoryOnly)
    if isFolder == 2:
        dialog.setFileMode(QFileDialog.AnyFile)
    # OPENING OR SAVING
    dialog.setAcceptMode(
        QFileDialog.AcceptOpen) if forOpen else dialog.setAcceptMode(
        QFileDialog.AcceptSave)

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
