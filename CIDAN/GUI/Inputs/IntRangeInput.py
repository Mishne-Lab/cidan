from qtpy.QtWidgets import *

from CIDAN.GUI.Inputs.Input import Input


class IntRangeInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip,
                 min, max, step, display_tool_tip=False
                 ):
        super().__init__(display_name, program_name, on_change_function, default_val,
                         tool_tip, display_tool_tip)

        self.input_box_1 = QSpinBox()
        self.input_box_1.setMinimum(min)
        self.input_box_1.setMaximumWidth(50)
        self.input_box_1.setMaximum(max)
        self.input_box_1.setSingleStep(step)
        self.input_box_1.setValue(self.default_val[0])
        self.input_box_1.setToolTip(self.tool_tip)
        self.layout_h.addWidget(self.input_box_1)
        self.layout_h.addWidget(QLabel(" to "))
        self.input_box_1.valueChanged.connect(self.on_change)
        self.input_box_2 = QSpinBox()
        self.input_box_2.setMinimum(min)
        self.input_box_2.setMaximumWidth(50)
        self.input_box_2.setMaximum(max)
        self.input_box_2.setSingleStep(step)
        self.input_box_2.setValue(self.default_val[1])
        self.input_box_2.setToolTip(self.tool_tip)
        self.layout_h.addWidget(self.input_box_2)
        self.input_box_2.valueChanged.connect(self.on_change)

    def current_state(self):
        return [self.input_box_1.value(), self.input_box_2.value()]

    def set_default_val(self):
        self.input_box_1.setValue(self.default_val)
        self.input_box_2.setValue(self.default_val)
