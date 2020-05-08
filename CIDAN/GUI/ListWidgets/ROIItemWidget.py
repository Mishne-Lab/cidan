from PySide2 import QtGui, QtCore
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class ROIItemWidget(QWidget):
    def __init__(self, roi_tab, color, roi_list, roi_num, parent=None):
        self.roi_tab = roi_tab
        self.roi_list = roi_list
        self.roi_num = roi_num
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
        self.zoom_button.clicked.connect(lambda x: self.roi_tab.zoomRoi(self.roi_num))
        # self.button = QtGui.QPushButton("Selection Add")
        # self.button.clicked.connect(lambda x: self.roi_tab.modify_roi(self.roi_num, "add"))
        # self.button2 = QtGui.QPushButton("Selection Subtract")
        # self.button2.clicked.connect(
        #     lambda x: self.roi_tab.modify_roi(self.roi_num, "subtract"))
        # self.button3 = QtGui.QPushButton("Delete ROI")
        self.check_box = QCheckBox()
        self.check_box.toggled.connect(lambda: self.check_box_toggled())
        self.check_box_time_trace = QCheckBox()
        self.check_box_time_trace.toggled.connect(lambda: self.time_check_box_toggled())
        out_img = QImage(100, 100, QImage.Format_ARGB32)
        out_img.fill(Qt.transparent)
        brush = QBrush(QColor(*color))  # Create texture brush
        painter = QPainter(out_img)  # Paint the output image
        painter.setBrush(brush)  # Use the image texture brush
        painter.setPen(Qt.NoPen)  # Don't draw an outline
        painter.setRenderHint(QPainter.Antialiasing, True)  # Use AA
        painter.drawEllipse(0, 0, 100, 100)  # Actually draw the circle
        painter.end()  # We are done (segfault if you forget this)

        # Convert the image to a pixmap and rescale it.  Take pixel ratio into
        # account to get a sharp image on retina displays:
        pr = QWindow().devicePixelRatio()
        pm = QPixmap.fromImage(out_img)
        label_pix = QLabel()
        label_pix.setPixmap(pm)
        label_pix.setMaximumWidth(10)
        lay = QtGui.QHBoxLayout(self)
        lay.addWidget(self.check_box, alignment=QtCore.Qt.AlignLeft)
        # lay.addWidget(label_pix, alignment=QtCore.Qt.AlignLeft)
        lay.addWidget(QLabel(text="#" + str(roi_num)), alignment=QtCore.Qt.AlignLeft)
        lay.addWidget(QLabel())
        lay.addWidget(QLabel())
        lay.addWidget(QLabel())
        lay.addWidget(QLabel())
        lay.addWidget(self.zoom_button, alignment=QtCore.Qt.AlignRight)
        lay.addWidget(self.check_box_time_trace, alignment=QtCore.Qt.AlignRight)
        # lay.addWidget(self.button, alignment=QtCore.Qt.AlignRight)
        # lay.addWidget(self.button2, alignment=QtCore.Qt.AlignRight)
        # lay.addWidget(self.button3, alignment=QtCore.Qt.AlignRight)
        lay.setContentsMargins(0, 0, 0, 0)

    def select_check_box(self):
        if not self.check_box.checkState():
            for x in self.roi_list.roi_item_list:
                if x != self:
                    x.check_box.setChecked(False)
            self.check_box.setChecked(True)
            self.roi_list.current_selected_roi = self.roi_num

        else:
            self.check_box.setChecked(False)
            self.roi_list.current_selected_roi = None

    def select_time_check_box(self):
        self.check_box_time_trace.setChecked(not self.check_box_time_trace.checkState())

    def check_box_toggled(self):
        if self.check_box.checkState():

            self.roi_tab.selectRoi(self.roi_num)

            for x in self.roi_list.roi_item_list:
                if x != self:
                    x.check_box.setChecked(False)

            self.roi_list.current_selected_roi = self.roi_num
        else:
            self.roi_list.current_selected_roi = None
            self.roi_tab.deselectRoi(self.roi_num)

    def time_check_box_toggled(self):
        self.roi_list.roi_time_check_list[
            self.roi_num - 1] = self.check_box_time_trace.checkState()
        if self.check_box_time_trace.checkState():
            self.roi_tab.selectRoiTime(self.roi_num)
        else:
            self.roi_tab.deselectRoiTime(self.roi_num)
