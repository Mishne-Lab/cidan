from qtpy import QtCore
from qtpy.QtWidgets import *


class ROIItemWidget(QWidget):
    """
    Item in the ROI list, takes care of everything except for color part which is
    handled by ROIItemModule
    """

    def __init__(self, roi_tab, color, roi_list, id, roi_num, parent=None,
                 display_time=True):
        self.roi_tab = roi_tab
        self.roi_list = roi_list
        self.display_time = display_time
        self.roi_num = roi_num
        self.id = id
        super(ROIItemWidget, self).__init__(parent)
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
        self.zoom_button = QPushButton("Zoom To")
        self.zoom_button.clicked.connect(
            lambda x: self.roi_tab.image_view.zoomRoi(self.roi_num))
        self.check_box = QCheckBox()
        self.check_box.toggled.connect(lambda: self.check_box_toggled())
        self.check_box_time_trace = QCheckBox()
        self.check_box_time_trace.toggled.connect(lambda: self.time_check_box_toggled())

        lay = QHBoxLayout(self)
        lay.addWidget(self.check_box, alignment=QtCore.Qt.AlignLeft)
        lay.addWidget(QLabel(text="#" + str(id)), alignment=QtCore.Qt.AlignLeft)
        if display_time:
            lay.addWidget(QLabel())
            lay.addWidget(QLabel())
            lay.addWidget(QLabel())
        # lay.addWidget(
        #     QLabel(str(round(self.roi_tab.data_handler.roi_circ_list[roi_num - 1], 3))))
        lay.addWidget(self.zoom_button)
        if display_time:
            lay.addWidget(self.check_box_time_trace, alignment=QtCore.Qt.AlignRight)
        lay.setContentsMargins(0, 0, 0, 0)

    def keyPressEvent(self, event):
        self.roi_tab.keyPressEvent(event)

    def select_check_box(self, force_on=False):
        if not self.check_box.checkState() or force_on:
            if not self.roi_list.select_multiple:
                for x in self.roi_list.roi_item_list:
                    if x != self:
                        x.check_box.setChecked(False)

            self.check_box.setChecked(True)

            if not self.display_time:
                self.check_box_time_trace.setChecked(True)
            self.roi_list.current_selected_roi = self.roi_num
            try:
                self.roi_tab.update_current_roi_selected()
            except AttributeError:
                pass

        else:
            self.check_box.setChecked(False)
            if not self.display_time:
                self.check_box_time_trace.setChecked(False)
            self.roi_list.current_selected_roi = None
            try:
                self.roi_tab.update_current_roi_selected()
            except AttributeError:
                pass

    def selected(self):
        return self.check_box.checkState()
    def select_time_check_box(self):
        self.check_box_time_trace.setChecked(not self.check_box_time_trace.checkState())

    def check_box_toggled(self):
        if self.check_box.checkState():

            if not self.roi_list.select_multiple:
                for x in self.roi_list.roi_item_list:
                    if x != self:
                        x.check_box.setChecked(False)

            self.roi_list.current_selected_roi = self.roi_num
            try:
                self.roi_tab.update_current_roi_selected()
            except AttributeError:
                pass
            self.check_box_time_trace.setChecked(True)

            if not self.display_time:
                self.roi_tab.image_view.selectRoi(self.roi_num)
        else:
            self.roi_list.current_selected_roi = None
            try:
                self.roi_tab.update_current_roi_selected()
            except AttributeError:
                pass
            if not self.display_time:
                self.check_box_time_trace.setChecked(False)
            self.roi_tab.image_view.deselectRoi(self.roi_num)

    def time_check_box_toggled(self):

        self.roi_list.roi_time_check_list[
            self.roi_num] = self.check_box_time_trace.checkState()
        try:
            if self.check_box_time_trace.checkState():
                self.roi_tab.selectRoiTime(self.roi_num)
            else:
                self.roi_tab.deselectRoiTime()
        except AttributeError:
            pass
