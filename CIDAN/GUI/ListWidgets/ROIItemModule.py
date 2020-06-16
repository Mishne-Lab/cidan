from qtpy import QtCore
from qtpy.QtGui import *


class ROIItemModule(QStandardItem):
    """
    An item in the ROI list, this part just takes care of the color part, the rest is
    handeled by roi item widget
    """
    def __init__(self, color, num, roi_tab):
        self.roi_tab = roi_tab
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
        self.num = num
        super().__init__(pm, "")
        self.setEditable(False)
        # self.setCheckable(True)

    def toggle_check_state(self):
        if not self.checkState():
            self.roi_tab.selectRoi(self.num)
            self.setCheckState(QtCore.Qt.CheckState.Checked)
        else:
            self.roi_tab.deselectRoi(self.num)
            self.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def checkState(self):
        state = super().checkState()
        if state == QtCore.Qt.CheckState.Unchecked:
            return False
        else:
            return True
