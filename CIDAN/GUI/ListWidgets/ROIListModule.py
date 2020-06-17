from qtpy import QtCore
from qtpy.QtGui import *
from qtpy.QtWidgets import *

from CIDAN.GUI.ListWidgets.ROIItemModule import ROIItemModule
from CIDAN.GUI.ListWidgets.ROIItemWidget import ROIItemWidget


class ROIListModule(QFrame):
    """
    Its the roi list in the ROI modification tab
    """

    def __init__(self, data_handler, roi_tab, select_multiple=False, display_time=True):
        super().__init__()
        self.current_selected_roi = 0
        self.select_multiple = select_multiple
        self.display_time = display_time
        self.roi_tab = roi_tab
        self.color_list = data_handler.color_list
        self.list = QListView()
        self.setStyleSheet("QListView::item { border-bottom: 1px solid rgb(50, 65, " +
                           "75); }")
        self.top_labels_layout = QHBoxLayout()
        label1 = QLabel(text="ROI Selected")
        label1.setMaximumWidth(100)
        self.top_labels_layout.addWidget(label1, alignment=QtCore.Qt.AlignRight)
        label2 = QLabel(text="ROI Num")
        label2.setMaximumWidth(100)
        self.top_labels_layout.addWidget(label2, alignment=QtCore.Qt.AlignLeft)
        self.top_labels_layout.addWidget(QLabel(), alignment=QtCore.Qt.AlignRight)
        self.top_labels_layout.addWidget(QLabel(), alignment=QtCore.Qt.AlignRight)
        self.top_labels_layout.addWidget(QLabel(), alignment=QtCore.Qt.AlignRight)
        self.top_labels_layout.addWidget(QLabel(), alignment=QtCore.Qt.AlignRight)
        if display_time:
            label3 = QLabel(text="Time Trace On")
            label3.setMaximumWidth(100)
            self.top_labels_layout.addWidget(label3, alignment=QtCore.Qt.AlignRight)
        self.model = QStandardItemModel(self.list)
        self.list.setModel(self.model)
        self.layout = QVBoxLayout()
        self.layout.addLayout(self.top_labels_layout)
        self.layout.addWidget(self.list)
        self.roi_item_list = []
        self.setLayout(self.layout)

    def set_current_select(self, num):
        self.list.setCurrentIndex(self.model.index(int(num - 1), 0))
        self.roi_time_check_list[num - 1] = not self.roi_time_check_list[num - 1]
        self.roi_item_list[num - 1].select_check_box()
        if self.display_time:
            self.roi_item_list[num - 1].select_time_check_box()

    def set_list_items(self, rois):
        self.roi_list = rois
        for x in range(self.model.rowCount()):
            self.model.removeRow(0)
        self.roi_item_list = []
        for num in range(len(self.roi_list)):
            item = ROIItemWidget(self.roi_tab,
                                 self.color_list[num % len(self.color_list)], self,
                                 num + 1, display_time=self.display_time
                                 )
            item1 = ROIItemModule(self.color_list[num % len(self.color_list)], num + 1,
                                  self.roi_tab)
            self.roi_item_list.append(item)
            self.model.appendRow(item1)
            self.list.setIndexWidget(item1.index(), item)
        self.roi_time_check_list = [False] * len(self.roi_item_list)
        if hasattr(self.roi_tab, "tab_selector_roi"):
            self.roi_tab.tab_selector_roi.setCurrentIndex(1)
        # self.roi_item_list[0].select_check_box()
        # self.roi_item_list[0].select_time_check_box()
    # def change(self):
    #     # This is a way of running the select roi function when a checkbox is clicked there
    #     # needed to be a work around because can't just connect a signal to it
    #     for num, item, check_val in zip(range(1,len(self.roi_time_check_list)+1),self.roi_item_list,self.roi_time_check_list):
    #         if item.checkState() != check_val:
    #             self.roi_time_check_list[num-1] = item.checkState()
    #             if item.checkState():
    #                 self.roi_tab.selectRoi(num)
    #             else:
    #                 self.roi_tab.deselectRoi(num)
