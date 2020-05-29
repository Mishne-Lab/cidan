from qtpy.QtWidgets import *

from CIDAN.GUI.Inputs.Input import Input


class IntInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip,
                 min, max, step, display_tool_tip=False
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
