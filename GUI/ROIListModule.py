from PySide2.QtWidgets import *
from PySide2.QtGui import *
from GUI.ROIItemModule import ROIItemModule
class ROIListModule(QFrame):
    def __init__(self, data_handler, roi_tab):
        super().__init__()
        self.roi_tab = roi_tab
        self.color_list = data_handler.color_list
        self.list = QListView()
        self.model = QStandardItemModel(self.list)
        self.model.itemChanged.connect(self.change)
        self.list.setModel(self.model)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.list)
        self.roi_item_list =[]
        self.setLayout(self.layout)
    def set_current_select(self,num):
        self.list.setCurrentIndex(self.model.index(int(num-1),0))
        self.roi_check_list[num - 1] = not self.roi_check_list[num - 1]
        self.roi_item_list[num-1].toggle_check_state()


    def set_list_items(self, clusters):
        self.cluster_list = clusters
        for x in range(self.model.rowCount()):
            self.model.removeRow(0)
        self.roi_item_list = []
        for num in range(len(self.cluster_list)):
            self.roi_item_list.append(ROIItemModule(self.color_list[num%len(self.color_list)],num+1,self.roi_tab))
            self.model.appendRow(self.roi_item_list[-1])
        self.roi_check_list = [False]*len(self.roi_item_list)
    def change(self):
        # This is a way of running the select roi function when a checkbox is clicked there
        # needed to be a work around because can't just connect a signal to it
        for num, item, check_val in zip(range(1,len(self.roi_check_list)+1),self.roi_item_list,self.roi_check_list):
            if item.checkState() != check_val:
                self.roi_check_list[num-1] = item.checkState()
                if item.checkState():
                    self.roi_tab.selectRoi(num)
                else:
                    self.roi_tab.deselectRoi(num)



