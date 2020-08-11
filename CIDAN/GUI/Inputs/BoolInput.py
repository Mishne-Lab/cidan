from qtpy.QtWidgets import *
from qtpy import QtCore
from CIDAN.GUI.Inputs.Input import Input


class BoolInput(Input):
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip, display_tool_tip=False):
        super().__init__(display_name, program_name, on_change_function, default_val,
                         tool_tip, display_tool_tip)

        self.input_box = QCheckBox()
        self.input_box.setMaximumWidth(40)
        print(self.logicalDpiX() / 96.0)
        self.input_box.setStyleSheet("""QCheckBox::indicator {
     width: %spx;
     height: %spx; }QCheckBox{margin %spx;
 }""" % (str(25*self.logicalDpiX() / 96.0),str(25*self.logicalDpiX() / 96.0),str(3*self.logicalDpiX() / 96.0)))

        self.input_box.setChecked(self.default_val)
        self.input_box.toggled.connect(self.on_change)
        self.input_box.setToolTip(self.tool_tip)
        self.layout_h.addWidget(self.input_box, alignment=QtCore.Qt.AlignCenter)

    def current_state(self):
        return self.input_box.isChecked()

    def set_default_val(self):
        self.input_box.setChecked(self.default_val)
