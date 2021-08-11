from qtpy.QtWidgets import *

from cidan.GUI.Inputs.Input import Input


class ColorInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip, name="Choose Color"):
        super().__init__(display_name, program_name, on_change_function,
                         default_val, tool_tip)

        self.color = default_val
        self.name = name

        self.current_color = QLabel()
        self.current_color.setStyleSheet("background-color: #32414B")
        self.current_color.setText(str(default_val))
        self.layout_h.addWidget(self.current_color, stretch=2)
        self.button = QPushButton()
        self.button.setText("Color Picker")
        self.button.clicked.connect(self.on_browse_button)
        self.button.setToolTip(tool_tip)
        self.layout_h.addWidget(self.button)

    def on_browse_button(self):
        color_temp = QColorDialog.getColor()

        if color_temp.isValid():
            self.color = (color_temp.red(), color_temp.green(), color_temp.blue())
            self.current_color.setText(str(self.color))
            if self.on_change_function is not None:
                self.on_change_function(self.color)

    def current_state(self):
        return self.color


def createFileDialog(directory='', forOpen=True, fmt='', isFolder=0,
                     name="Choose Dataset:"):
    directory = "/Users/sschickler/Documents/LSSC-python/input_images"
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    options |= QFileDialog.DontUseCustomDirectoryIcons
    dialog = QFileDialog()
    dialog.setOptions(options)
    dialog.setWindowTitle(name)
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
