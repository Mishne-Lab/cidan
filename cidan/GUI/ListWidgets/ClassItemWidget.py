from qtpy import QtCore
from qtpy.QtWidgets import *


class ClassItemWidget(QWidget):
    """
    Item in the Class list, takes care of everything except for color part which is
    handled by ClassItemModule
    """

    def __init__(self, classifier_tab, color, class_dict, class_num, name, parent=None,
                 id=id):
        self.classifier_tab = classifier_tab
        self.class_dict = class_dict
        self.name = name
        self.id = id
        self.class_num = class_num
        super(ClassItemWidget, self).__init__(parent)
        self.setStyleSheet("""QPushButton {background-color: rgba(0,0,0,0%);
        padding-left:3px;
        padding-right:3px;
        
        color: #CCCCCC;}
        QPushButton:hover {
          border: 1px solid #148CD2;
          background-color: #505F69;
          color: #F0F0F0;
            }
            QPushButton:pressed {
              background-color: #19232D;
              border: 1px solid #19232D;
            }
            
            QPushButton:pressed:hover {
              border: 1px solid #148CD2;
            }
            QPushButton:selected {
              background-color: rgba(0,0,0,0%);
              color: #32414B;
            }
            QLabel {
            background-color: rgba(0,0,0,0%)
            }QCheckBox {
            background-color: rgba(0,0,0,0%)
            }""")
        self.edit_class = QPushButton("Edit Class")
        self.edit_class.clicked.connect(
            lambda x: self.classifier_tab.edit_class(self.id))
        # self.check_box = QCheckBox()
        # self.check_box.toggled.connect(lambda: self.check_box_toggled())
        # self.check_box_time_trace = QCheckBox()
        # self.check_box_time_trace.toggled.connect(lambda: self.time_check_box_toggled())

        lay = QHBoxLayout(self)
        lay.addWidget(QLabel(self.name), alignment=QtCore.Qt.AlignLeft)
        lay.addWidget(QLabel(text="#" + str(class_num)), alignment=QtCore.Qt.AlignLeft)
        # if display_time:
        lay.addWidget(QLabel())
        lay.addWidget(QLabel())
        lay.addWidget(QLabel())
        # lay.addWidget(
        #     QLabel(str(round(self.classifier_tab.data_handler.class_circ_list[class_num - 1], 3))))
        lay.addWidget(self.edit_class)
        lay.setContentsMargins(0, 0, 0, 0)

    def keyPressEvent(self, event):
        self.classifier_tab.keyPressEvent(event)
    # def select_check_box(self):
    #     if not self.check_box.checkState():
    #         if not self.class_dict.select_multiple:
    #             for x in self.class_dict.class_item_list:
    #                 if x != self:
    #                     x.check_box.setChecked(False)
    #         self.check_box.setChecked(True)
    #         if not self.display_time:
    #             self.check_box_time_trace.setChecked(True)
    #         self.class_dict.current_selected_class = self.class_num
    #
    #     else:
    #         self.check_box.setChecked(False)
    #         if not self.display_time:
    #             self.check_box_time_trace.setChecked(False)
    #         self.class_dict.current_selected_class = None
    #
    # def selected(self):
    #     return self.check_box.checkState()
    # def select_time_check_box(self):
    #     self.check_box_time_trace.setChecked(not self.check_box_time_trace.checkState())
    #
    # def check_box_toggled(self):
    #     if self.check_box.checkState():
    #
    #         if not self.class_dict.select_multiple:
    #             for x in self.class_dict.class_item_list:
    #                 if x != self:
    #                     x.check_box.setChecked(False)
    #
    #         self.class_dict.current_selected_class = self.class_num
    #         self.check_box_time_trace.setChecked(True)
    #
    #         if not self.display_time:
    #             self.classifier_tab.image_view.selectRoi(self.class_num)
    #     else:
    #         self.class_dict.current_selected_class = None
    #         if not self.display_time:
    #             self.check_box_time_trace.setChecked(False)
    #         self.classifier_tab.image_view.deselectRoi(self.class_num)
    #
    # def time_check_box_toggled(self):
    #
    #     self.class_dict.class_time_check_list[
    #         self.class_num - 1] = self.check_box_time_trace.checkState()
    #     if self.check_box_time_trace.checkState():
    #         self.classifier_tab.selectRoiTime(self.class_num)
    #     else:
    #         self.classifier_tab.deselectRoiTime()
