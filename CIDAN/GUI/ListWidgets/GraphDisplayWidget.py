from qtpy.QtGui import *
import numpy as np
from qtpy.QtGui import *
from qtpy.QtWidgets import *

from CIDAN.GUI.ListWidgets.GraphItemLine import GraphItemLine
from CIDAN.GUI.ListWidgets.GraphItemPColor import GraphItemPColor


class GraphItemStandard(QStandardItem):
    """
    An item in the ROI list, this part just takes care of the color part, the rest is
    handeled by roi item widget
    """
    # def __init__(self):


class GraphDisplayWidget(QFrame):
    def __init__(self, main_widget):
        super().__init__()
        self.current_selected_roi = 0
        self.main = main_widget

        self.list = QListView()
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list.verticalScrollBar().setSingleStep(7)
        self.list.setMinimumHeight(400)
        self.setStyleSheet("QListView::item { border-bottom: 1px solid rgb(50, 65, " +
                           "75); min-height: 100px;}"
                           "QListView::item:selected {background: rgb(25,35,45);}"
                           "QListView::item:hover {background: rgb(25,35,45);}")

        self.model = QStandardItemModel(self.list)
        self.list.setModel(self.model)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.list)
        self.roi_item_list = []
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 00, 0)
        self.p_color = True
        self.items = []
        self.loaded_rois = []

    def set_list_items(self, data_list, roi_names, trial_names, p_color=True,
                       type="neuron"):

        if (type == "neuron"):

            if p_color:
                for x in range(self.model.rowCount()):
                    self.model.removeRow(0)
                if len(data_list) > 0:
                    data = np.vstack(data_list)
                    item = GraphItemPColor(data=data, x_label="Time(Slices)",
                                           y_label="ROIs", x_ticks=None,
                                           y_ticks=roi_names, display_y_axis_ticks=True)
                    item1 = QStandardItem()
                    self.items = [item]
                    self.loaded_rois = roi_names
                    self.setStyleSheet(
                        "QListView::item { border-bottom: 1px solid rgb(50, 65, " +
                        "75); min-height: %s px;}" % str(
                            len(data_list) * 45 if len(data_list) > 5 else 220) +
                        "QListView::item:selected {background: rgb(25,35,45);}" +
                        "QListView::item:hover {background: rgb(25,35,45);}")
                    self.model.appendRow(item1)
                    self.list.setIndexWidget(item1.index(), item)

                    # for num, data, roi_num in zip(range(len(data_list)),data_list, roi_names):
            else:
                if p_color == self.p_color:
                    for x in list(self.loaded_rois):
                        if x not in roi_names:
                            self.items.pop(self.loaded_rois.index(x))
                            self.model.removeRow(self.loaded_rois.index(x))
                            self.loaded_rois.remove(x)
                else:
                    for x in range(self.model.rowCount()):
                        self.model.removeRow(0)
                    self.items = []
                    self.loaded_rois = []
                for data, name in zip(data_list, roi_names):
                    if name in self.loaded_rois and \
                            self.items[self.loaded_rois.index(name)].data[0] == data[0]:
                        pass
                    else:
                        item = GraphItemLine(data=data, x_label="Time(Slices)",
                                             y_label="ROI %s" % str(name), x_ticks=None,
                                             y_ticks=None,
                                             display_y_axis_ticks=True)

                        item1 = QStandardItem()
                        index = 0
                        while index < len(self.loaded_rois):
                            if name > self.loaded_rois[index]:
                                break
                            if name == self.loaded_rois[index]:
                                self.items.remove(index)
                                self.loaded_rois.pop(index)
                                self.model.removeRow(index)
                                break
                            index += 1
                        self.items.insert(index, item)
                        self.loaded_rois.insert(index, name)
                        self.setStyleSheet(
                            "QListView::item { border-bottom: 1px solid rgb(50, 65, " +
                            "75); min-height: %s px;}" % str(85) +
                            "QListView::item:selected {background: rgb(25,35,45);}" +
                            "QListView::item:hover {background: rgb(25,35,45);}")
                        self.model.insertRow(index, item1)
                        self.list.setIndexWidget(item1.index(), item)
            self.p_color = p_color
