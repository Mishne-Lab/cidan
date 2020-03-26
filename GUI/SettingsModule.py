from PySide2.QtWidgets import *
from GUI.Module import Module
import sys
import qdarkstyle
from GUI.SettingBlockModule import SettingBlockModule, filter_setting_block, \
    dataset_setting_block

class SettingsModule(Module):
    def __init__(self, importance, name, *args, show_name=True):
        super().__init__(importance)
        self.setMaximumWidth(300)
        self.setStyleSheet("SettingsModule {margin:5px; border:1px solid rgb(50, 65, "
                           "75);} ")
        self.setting_block_list = args
        self.layout = QVBoxLayout()
        if show_name:
            self.header = QLabel()
            self.header.setText(name)
            self.header.setStyleSheet("font-size: 20px")
            self.layout.addWidget(self.header)
        self.setting_block_layout = QToolBox()
        for block in self.setting_block_list:
            self.setting_block_layout.addItem(block, block.name)
        self.layout.addWidget(self.setting_block_layout)
        self.setLayout(self.layout)
def preprocessing_settings(main_widget):
    return SettingsModule(1, "Preprocessing Settings", dataset_setting_block(main_widget),
                          filter_setting_block(main_widget))
if __name__ == "__main__":
    app = QApplication([])

    widget = preprocessing_settings()
    widget.setStyleSheet(qdarkstyle.load_stylesheet())
    widget.show()


    sys.exit(app.exec_())