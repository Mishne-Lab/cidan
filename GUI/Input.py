from GUI.Module import Module
from PySide2.QtWidgets import *
from PySide2 import QtGui
from GUI.fileHandling import createFileDialog


class Input(Module):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip):
        super().__init__(1)
        self.program_name = program_name
        self.input_box = None
        self.tool_tip = tool_tip
        self.default_val = default_val
        self.display_name = display_name
        self.layout = QHBoxLayout()
        temp_lable = QLabel()
        temp_lable.setText(display_name)
        temp_lable.setToolTip(tool_tip)
        self.layout.addWidget(temp_lable)
        self.on_change_function = lambda x: on_change_function(program_name, x)
        self.setLayout(self.layout)

    def on_change(self, *args, **kwargs):
        self.on_change_function(self.program_name, self.current_state())

    def current_state(self):
        pass


class FloatInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip,
                 min, max, step,
                 ):
        super().__init__(display_name, program_name, on_change_function, default_val,
                         tool_tip)

        self.input_box = QDoubleSpinBox()
        self.input_box.setMinimum(min)
        self.input_box.setMaximum(max)
        self.input_box.setMaximumWidth(50)
        self.input_box.setSingleStep(step)
        self.input_box.setValue(self.default_val)
        self.input_box.valueChanged.connect(self.on_change_function)
        self.input_box.setToolTip(self.tool_tip)
        self.layout.addWidget(self.input_box)

    def current_state(self):
        return self.input_box.value()


class IntInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip,
                 min, max, step,
                 ):
        super().__init__(display_name, program_name, on_change_function, default_val,
                         tool_tip)

        self.input_box = QSpinBox()
        self.input_box.setMinimum(min)
        self.input_box.setMaximumWidth(50)
        self.input_box.setMaximum(max)
        self.input_box.setSingleStep(step)
        self.input_box.setValue(self.default_val)
        self.input_box.setToolTip(self.tool_tip)
        self.layout.addWidget(self.input_box)
        self.input_box.valueChanged.connect(self.on_change_function)

    def current_state(self):
        return self.input_box.value()


class BoolInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip):
        super().__init__(display_name, program_name, on_change_function, default_val,
                         tool_tip)

        self.input_box = QRadioButton()
        self.input_box.setMaximumWidth(50)
        self.input_box.setChecked(self.default_val)
        self.input_box.toggled.connect(self.on_change_function)
        self.input_box.setToolTip(self.tool_tip)
        self.layout.addWidget(self.input_box)

    def current_state(self):
        return self.input_box.isChecked()


class OptionInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_index,
                 tool_tip,
                 val_list):
        super().__init__(display_name, program_name, on_change_function,
                         default_index, tool_tip)

        self.input_box = QComboBox()
        self.input_box.addItems(val_list)
        self.input_box.setMaximumWidth(50)
        self.input_box.setCurrentIndex(self.default_val)
        self.input_box.setToolTip(self.tool_tip)
        self.input_box.currentIndexChanged.connect(self.on_change_function)
        self.layout.addWidget(self.input_box)

    def current_state(self):
        return self.input_box.isChecked()


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
        self.layout.addWidget(self.current_location)
        self.button = QPushButton()
        self.button.setText("Browse")
        self.button.clicked.connect(self.on_browse_button)
        self.button.setToolTip(tool_tip)
        self.layout.addWidget(self.button)

    def on_browse_button(self):
        self.path = createFileDialog(directory="~/Desktop", forOpen=self.forOpen,
                                     isFolder=self.isFolder)
        self.current_location.setText(self.path)

    def current_state(self):
        return self.path
