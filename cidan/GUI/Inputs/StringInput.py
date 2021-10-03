from qtpy.QtWidgets import *

from cidan.GUI.Inputs.Input import Input


class StringInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip,
                 display_tool_tip=False
                 ):
        super().__init__(display_name, program_name, on_change_function, default_val,
                         tool_tip, display_tool_tip)

        self.input_box = QLineEdit()
        # self.input_box.setMinimum(min)
        # self.input_box.setMaximumWidth(75*((self.logicalDpiX() / 96.0-1)/2+1))
        self.input_box.insert(self.default_val)
        self.input_box.setToolTip(self.tool_tip)
        self.layout_h.addWidget(self.input_box)
        self.input_box.textChanged.connect(self.on_change)

    def current_state(self):
        return self.input_box.text()

    def set_default_val(self):
        self.input_box.insert(self.default_val)
