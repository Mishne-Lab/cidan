from qtpy import QtCore
from qtpy.QtGui import *
from qtpy.QtWidgets import *

from cidan.GUI.ListWidgets.ClassItemModule import ClassItemModule
from cidan.GUI.ListWidgets.ClassItemWidget import ClassItemWidget


class ClassListModule(QFrame):
    """
    Its the class list in the Class modification tab
    """

    def __init__(self, data_handler, classifier_tab, select_multiple=False,
                 display_time=True):
        super().__init__()
        self.current_selected_class = 0
        self.select_multiple = select_multiple
        self.display_time = display_time
        self.classifier_tab = classifier_tab
        self.color_list = data_handler.color_list
        self.list = QListView()

        self.setStyleSheet("QListView::item { border-bottom: 1px solid rgb(50, 65, " +
                           "75); } " + """QListView::item::pressed,
 {
  background-color: #19232D;
  border: 1px solid #32414B;
  color: #F0F0F0;
  gridline-color: #32414B;
  border-radius: 4px;
}""")
        self.top_labels_layout = QHBoxLayout()
        self.top_labels_layout.setContentsMargins(0, 0, 0, 0)
        label1 = QLabel(text="Class")
        label1.setMaximumWidth(170 * ((self.logicalDpiX() / 96.0)))
        self.top_labels_layout.addWidget(label1, alignment=QtCore.Qt.AlignRight)
        label2 = QLabel(text="Class num")
        label2.setMaximumWidth(140 * ((self.logicalDpiX() / 96.0)))
        self.top_labels_layout.addWidget(label2, alignment=QtCore.Qt.AlignLeft)
        self.top_labels_layout.addWidget(QLabel(), alignment=QtCore.Qt.AlignRight)
        self.top_labels_layout.addWidget(QLabel(), alignment=QtCore.Qt.AlignRight)
        self.top_labels_layout.addWidget(QLabel(), alignment=QtCore.Qt.AlignRight)
        self.top_labels_layout.addWidget(QLabel(), alignment=QtCore.Qt.AlignRight)

        self.setMinimumHeight(200 * ((self.logicalDpiX() / 96.0 - 1) / 2 + 1))
        self.model = QStandardItemModel(self.list)
        self.list.setModel(self.model)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addLayout(self.top_labels_layout)
        self.layout.addWidget(self.list)
        self.class_item_list = []
        self.setLayout(self.layout)

    def keyPressEvent(self, event):
        self.classifier_tab.keyPressEvent(event)

    # def set_current_select(self, num):
    #     self.list.setCurrentIndex(self.model.index(int(num - 1), 0))
    #     self.class_time_check_list[num - 1] = not self.class_time_check_list[num - 1]
    #     self.class_item_list[num - 1].select_check_box()
    #     # self.class_module_list[num-1].
    #     if self.display_time:
    #         self.class_item_list[num - 1].select_time_check_box()

    def set_list_items(self, class_dict):
        self.class_dict = class_dict
        for x in range(self.model.rowCount()):
            self.model.removeRow(0)
        self.class_item_list = []
        self.class_module_list = []
        for num, id in enumerate(list(self.class_dict.keys())):
            item = ClassItemWidget(self.classifier_tab,
                                   self.class_dict[id]["color"], self,
                                   num + 1, name=self.class_dict[id]["name"], id=id
                                   )
            item1 = ClassItemModule(self.class_dict[id]["color"], num + 1,
                                    name=self.class_dict[id]["name"],
                                    id=id, classifier_tab=self.classifier_tab)
            item1.setSelectable(False)
            self.class_item_list.append(item)
            self.model.appendRow(item1)
            self.class_module_list.append(item1)
            self.list.setIndexWidget(item1.index(), item)

        # self.class_item_list[0].select_check_box()
        # self.class_item_list[0].select_time_check_box()
    # def change(self):
    #     # This is a way of running the select class function when a checkbox is clicked there
    #     # needed to be a work around because can't just connect a signal to it
    #     for num, item, check_val in zip(range(1,len(self.class_time_check_list)+1),self.class_item_list,self.class_time_check_list):
    #         if item.checkState() != check_val:
    #             self.class_time_check_list[num-1] = item.checkState()
    #             if item.checkState():
    #                 self.classifier_tab.selectRoi(num)
    #             else:
    #                 self.classifier_tab.deselectRoi(num)
