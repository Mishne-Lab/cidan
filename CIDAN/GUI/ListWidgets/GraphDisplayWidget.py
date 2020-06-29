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


class GraphDisplayWidget(QWidget):
    def __init__(self, main_widget):
        super().__init__()
        self.current_selected_roi = 0
        self.setMinimumHeight(200)
        self.main_widget = main_widget
        self.color_list = self.data_handler.color_list
        self.current_graph = QWidget()
        self.layout = QHBoxLayout()
        self.layout.addWidget(self.current_graph)
        self.roi_item_list = []
        self.setLayout(self.layout)
        self.layout.setContentsMargins(0, 0, 00, 0)
        self.p_color = True
        self.std = 5

    @property
    def data_handler(self):
        return self.main_widget.data_handler
    def set_list_items(self, data_list, roi_names, trial_names, p_color=True,
                       type="neuron"):
        try:
            self.layout.removeWidget(self.current_graph)
            self.current_graph.deleteLater()
        except RuntimeError:
            pass

        if (type == "neuron"):

            if p_color:

                if len(data_list) > 0:
                    data = np.vstack(data_list)
                    self.current_graph = GraphItemPColor(data=data,
                                                         x_label="Time(Slices)",
                                                         y_label="ROIs", x_ticks=None,
                                                         y_ticks=roi_names, display_y_axis_ticks=True)
                    self.layout.addWidget(self.current_graph)


                    # for num, data, roi_num in zip(range(len(data_list)),data_list, roi_names):
                else:
                    self.current_graph = QWidget()
                    self.layout.addWidget(self.current_graph)
            else:

                if len(data_list) > 0:
                    data_list_processed = [
                        ((x - np.mean(x)) / (np.std(x) * self.std)) + num for num, x in
                        enumerate(data_list)]
                    # data = np.vstack(data_list_processed)
                    self.current_graph = GraphItemLine(data=data_list_processed,
                                                       x_label="Time(Slices)",
                                                       y_label="ROI", x_ticks=None,
                                                       y_ticks=roi_names,
                                                       display_y_axis_ticks=True,
                                                       display_roi_labels=True,
                                                       roi_labels=roi_names, colors=[
                            self.color_list[x % len(self.color_list)] for x in
                            roi_names])
                    self.layout.addWidget(self.current_graph)
                else:
                    self.current_graph = QWidget()
                    self.layout.addWidget(self.current_graph)

            self.p_color = p_color
        if (type == "trial"):
            if p_color:

                if len(data_list) > 0:
                    median_length = np.median([x.shape[0] for x in data_list])
                    data_list = [np.pad(x, [(0,int(median_length-x.shape[0]))], mode="constant") for x in data_list]
                    data = np.vstack(data_list)
                    self.current_graph = GraphItemPColor(data=data,
                                                         x_label="Time(For ROI %s)" % str(
                                                             roi_names[0]),
                                                         y_label="Trial", x_ticks=None,
                                                         y_ticks=trial_names,
                                                         display_y_axis_ticks=True)
                    self.layout.addWidget(self.current_graph)

                    # for num, data, roi_num in zip(range(len(data_list)),data_list, roi_names):
                else:
                    self.current_graph = QWidget()
                    self.layout.addWidget(self.current_graph)
            else:

                if len(data_list) > 0:
                    data_list_processed = [
                        (((x - np.mean(x)) / (np.std(x) * self.std)) + num) for num, x
                        in
                        enumerate(data_list)]
                    # data = np.vstack(data_list_processed)
                    self.current_graph = GraphItemLine(data=data_list_processed,
                                                       x_label="Time(For ROI %s)" % str(
                                                           roi_names[0]),
                                                       y_label="Trial", x_ticks=None,
                                                       y_ticks=trial_names,
                                                       display_y_axis_ticks=True,
                                                       display_roi_labels=True,
                                                       roi_labels=trial_names, colors=[
                            self.color_list[x % len(self.color_list)] for x in
                            range(len(trial_names))])
                    self.layout.addWidget(self.current_graph)
                else:
                    self.current_graph = QWidget()
                    self.layout.addWidget(self.current_graph)

            self.p_color = p_color
