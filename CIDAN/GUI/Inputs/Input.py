from qtpy.QtWidgets import *


class Input(QFrame):
    """
    Default Input class implemeneted by all other input classes, all have same args and
    functions
    """
    def __init__(self, display_name, program_name, on_change_function, default_val,
                 tool_tip, display_tool_tip=False, show_name=True):
        super().__init__()
        self.program_name = program_name
        self.input_box_1 = None
        self.tool_tip = tool_tip
        self.default_val = default_val
        self.display_name = display_name
        self.layout_main = QVBoxLayout()
        self.layout_main.setContentsMargins(2, 2, 2, 2)
        self.layout_h = QHBoxLayout()
        self.layout_h.setContentsMargins(0, 0, 0, 0)
        if show_name:
            temp_label = QLabel()
            temp_label.setText(display_name)
            temp_label.setToolTip(tool_tip)
            self.layout_h.addWidget(temp_label)
        self.on_change_function = on_change_function
        self.layout_main.addLayout(self.layout_h)
        if display_tool_tip:
            temp_lable_2 = QLabel()
            temp_lable_2.setText(tool_tip)
            self.layout_main.addWidget(temp_lable_2)

        self.setLayout(self.layout_main)

    def on_change(self, *args, **kwargs):
        try:
            if (not self.on_change_function == None):
                self.on_change_function(self.program_name, self.current_state())
        except AssertionError as e:
            print(e)
            self.set_default_val()

    def current_state(self):
        pass

    def set_default_val(self):
        pass

# TODO implement way to check spatial boxes square
