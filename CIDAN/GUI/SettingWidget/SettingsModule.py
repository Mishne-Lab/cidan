import sys

import qdarkstyle

from CIDAN.GUI.SettingWidget.SettingBlockModule import *


class SettingsModule(QFrame):
    """
    A Module that contains multiple SettingBlockModule each displayed as a tab
    """
    def __init__(self, name, *args, show_name=True):
        super().__init__()

        self.setting_block_list = args
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(2, 2, 2, 2)
        if show_name:
            # self.setStyleSheet("SettingsModule { border:1px solid rgb(50, 65, "
            #                    "75);} ")
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
    return SettingsModule("Preprocessing Settings",
                          filter_setting_block(main_widget),
                          dataset_setting_block_crop(main_widget),
                          dataset_setting_block(main_widget))


def roi_extraction_settings(main_widget):
    return SettingsModule("ROI Extraction Settings",
                          multiprocessing_settings_block(main_widget),
                          roi_extraction_settings_block(main_widget),
                          roi_advanced_settings_block(main_widget), show_name=False)


if __name__ == "__main__":
    app = QApplication([])

    widget = preprocessing_settings()  # noqa
    widget.setStyleSheet(qdarkstyle.load_stylesheet())
    widget.show()

    sys.exit(app.exec_())
