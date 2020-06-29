from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import *


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, edgecolor=(1, 1, 1))
        self.fig.patch.set_facecolor((.098, .13725, .17647))
        self.axes = self.fig.add_subplot()
        self.fig.tight_layout(pad=0)
        self.axes.tick_params(colors='white')
        super(MplCanvas, self).__init__(self.fig)


class GraphItemPColor(QWidget):
    def __init__(self, *, data, x_label, y_label, x_ticks, y_ticks,
                 display_y_axis_ticks=True, display_axis_labels=True, aspect_ratio_x=9,
                 aspect_ratio_y=1):
        super(GraphItemPColor, self).__init__()
        self.setStyleSheet("""
                                QLabel {
                                background-color: rgba(0,0,0,0%)
                                }""")
        layout = QVBoxLayout()
        layout_h = QHBoxLayout()
        self.data = data
        self.y_label = y_label
        if len(data.shape) == 1:
            data = data.reshape((-1, 1))
            data = data.transpose((1, 0))
        self.graph = MplCanvas(width=aspect_ratio_x, height=aspect_ratio_y)
        self.graph.axes.pcolor(data, cmap='jet')

        # self.graph.axes.xticks(ticks=list(range(data.shape[0])), labels=x_ticks)
        if display_y_axis_ticks and y_ticks != None:
            self.graph.axes.yaxis.set_ticks(ticks=list(range(data.shape[0])))
            self.graph.axes.set_yticklabels(labels=y_ticks)
        if not display_y_axis_ticks:
            self.graph.axes.yaxis.set_ticks([])
        self.graph.axes.set_xlabel(x_label, color="white")
        # self.graph.axes.set_ylabel(y_label, color="white")
        layout_h.addWidget(QLabel(y_label))
        # test = QLabel("")
        # test.setMinimumHeight(200)
        # layout_h.addWidget(test)
        layout_h.addWidget(self.graph)

        layout.addLayout(layout_h, stretch=10)
        # layout.addWidget(QLabel(x_label), alignment=QtCore.Qt.AlignCenter,stretch=1)

        self.setLayout(layout)
        # self.setMinimumHeight(200)
        layout.setContentsMargins(0, 0, 10, 0)
        layout_h.setContentsMargins(0, 0, 00, 0)
