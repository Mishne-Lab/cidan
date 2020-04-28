from PySide2.QtWidgets import *
from PySide2 import QtGui
from CIDAN.fileHandling import createFileDialog


class Input(QFrame):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip, display_tool_tip =False):
        super().__init__()
        self.program_name = program_name
        self.input_box_1 = None
        self.tool_tip = tool_tip
        self.default_val = default_val
        self.display_name = display_name
        self.layout_main = QVBoxLayout()
        self.layout_h = QHBoxLayout()
        temp_lable = QLabel()
        temp_lable.setText(display_name)
        temp_lable.setToolTip(tool_tip)
        self.layout_h.addWidget(temp_lable)
        self.on_change_function = on_change_function
        self.layout_main.addLayout(self.layout_h)
        if display_tool_tip:
            temp_lable_2 = QLabel()
            temp_lable_2.setText(tool_tip)
            self.layout_main.addWidget(temp_lable_2)

        self.setLayout(self.layout_main)

    def on_change(self, *args, **kwargs):
        try:
            self.on_change_function(self.program_name, self.current_state())
        except AssertionError as e:
            print(e)
            self.set_default_val()

    def current_state(self):
        pass
    def set_default_val(self):
        pass

class FloatInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip,
                 min, max, step, display_tool_tip =False
                 ):
        super().__init__(display_name, program_name, on_change_function, default_val,
                         tool_tip, display_tool_tip)

        self.input_box = QDoubleSpinBox()
        self.input_box.setMinimum(min)
        self.input_box.setMaximum(max)
        self.input_box.setMaximumWidth(50)
        self.input_box.setSingleStep(step)
        self.input_box.setValue(self.default_val)
        self.input_box.valueChanged.connect(self.on_change)
        self.input_box.setToolTip(self.tool_tip)
        self.layout_h.addWidget(self.input_box)

    def current_state(self):
        return self.input_box.value()
    def set_default_val(self):
        self.input_box.setValue(self.default_val)


class IntInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip,
                 min, max, step,  display_tool_tip =False
                 ):
        super().__init__(display_name, program_name, on_change_function, default_val,
                         tool_tip, display_tool_tip)

        self.input_box = QSpinBox()
        self.input_box.setMinimum(min)
        self.input_box.setMaximumWidth(50)
        self.input_box.setMaximum(max)
        self.input_box.setSingleStep(step)
        self.input_box.setValue(self.default_val)
        self.input_box.setToolTip(self.tool_tip)
        self.layout_h.addWidget(self.input_box)
        self.input_box.valueChanged.connect(self.on_change)

    def current_state(self):
        return self.input_box.value()
    def set_default_val(self):
        self.input_box.setValue(self.default_val)


# TODO implement way to check spatial boxes square
class BoolInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip, display_tool_tip =False):
        super().__init__(display_name, program_name, on_change_function, default_val,
                         tool_tip, display_tool_tip)

        self.input_box = QRadioButton()
        self.input_box.setMaximumWidth(50)
        self.input_box.setChecked(self.default_val)
        self.input_box.toggled.connect(self.on_change)
        self.input_box.setToolTip(self.tool_tip)
        self.layout_h.addWidget(self.input_box)

    def current_state(self):
        return self.input_box.isChecked()
    def set_default_val(self):
        self.input_box.setChecked(self.default_val)

class OptionInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_index,
                 tool_tip,
                 val_list, display_tool_tip =False):
        super().__init__(display_name, program_name, on_change_function,
                         default_index, tool_tip, display_tool_tip)

        self.input_box = QComboBox()
        self.val_list = val_list
        self.input_box.addItems(val_list)

        self.input_box.setCurrentIndex(self.default_val)
        self.input_box.setToolTip(self.tool_tip)
        self.input_box.currentIndexChanged.connect(self.on_change)
        self.layout_h.addWidget(self.input_box)

    def current_state(self):
        return self.val_list[self.input_box.currentIndex()]
    def set_default_val(self):
        self.input_box.setCurrentIndex(self.default_val)



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
        if self.on_change_function != None:
            self.on_change_function(self.path)

    def current_state(self):
        return self.path
class Int3DInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip,
                 min, max, step,  display_tool_tip =False
                 ):
        super().__init__(display_name, program_name, on_change_function, default_val,
                         tool_tip, display_tool_tip)

        self.input_box_1 = QSpinBox()
        self.input_box_1.setMinimum(min)
        self.input_box_1.setMaximumWidth(50)
        self.input_box_1.setMaximum(max)
        self.input_box_1.setSingleStep(step)
        self.input_box_1.setValue(self.default_val)
        self.input_box_1.setToolTip(self.tool_tip)
        self.layout_h.addWidget(self.input_box_1)
        self.input_box_1.valueChanged.connect(self.on_change)
        self.input_box_2 = QSpinBox()
        self.input_box_2.setMinimum(min)
        self.input_box_2.setMaximumWidth(50)
        self.input_box_2.setMaximum(max)
        self.input_box_2.setSingleStep(step)
        self.input_box_2.setValue(self.default_val)
        self.input_box_2.setToolTip(self.tool_tip)
        self.layout_h.addWidget(self.input_box_2)
        self.input_box_2.valueChanged.connect(self.on_change)
        self.input_box_3 = QSpinBox()
        self.input_box_3.setMinimum(min)
        self.input_box_3.setMaximumWidth(50)
        self.input_box_3.setMaximum(max)
        self.input_box_3.setSingleStep(step)
        self.input_box_3.setValue(self.default_val)
        self.input_box_3.setToolTip(self.tool_tip)
        self.layout_h.addWidget(self.input_box_3)
        self.input_box_3.valueChanged.connect(self.on_change)

    def current_state(self):
        return self.input_box_1.value()
    def set_default_val(self):
        self.input_box_1.setValue(self.default_val)
