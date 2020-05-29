from qtpy.QtWidgets import *

from CIDAN.GUI.Inputs.Input import Input


class OptionInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_index,
                 tool_tip,
                 val_list, display_tool_tip=False, show_name=True):
        super().__init__(display_name, program_name, on_change_function,
                         default_index, tool_tip, display_tool_tip, show_name=show_name)

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
