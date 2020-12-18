import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtWidgets import *


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, edgecolor=(1, 1, 1))
        fig.patch.set_facecolor((.098, .13725, .17647))
        self.axes = fig.add_subplot()
        fig.tight_layout(pad=0)
        self.axes.tick_params(colors='white')
        super(MplCanvas, self).__init__(fig)


class GraphItemLine(QWidget):
    def __init__(self, *, data, x_label, y_label, x_ticks, y_ticks,
                 display_y_axis_ticks=True, display_axis_labels=True, aspect_ratio_x=9,
                 display_roi_labels=False, roi_labels=[],
                 aspect_ratio_y=1, colors=False):
        super(GraphItemLine, self).__init__()

        self.setStyleSheet("""
                                QLabel {
                                background-color: rgba(0,0,0,0%)
                                }""")
        layout = QVBoxLayout()
        layout_h = QHBoxLayout()

        self.data = data
        self.y_label = y_label
        # if len(data.shape) ==1:
        # data = data.reshape((-1, 1))
        # data = data.transpose((1,0))
        self.graph = pg.PlotWidget()

        self.graph.showGrid(x=True, y=True, alpha=0.3)
        if (type(data) == list):
            if type(colors) == list:
                for x, color in zip(data, colors):
                    pen = pg.mkPen(color=color, width=2)
                    self.graph.plot(x, pen=pen)
            else:
                for x in data:
                    self.graph.plot(x)
            if (display_roi_labels):
                for num, name in enumerate(roi_labels):
                    text_box = pg.TextItem(y_label + "_" + str(name), anchor=(0, .5),
                                           fill=pg.mkBrush(color=(0, 0, 0)))

                    self.graph.addItem(text_box)
                    text_box.setPos(-5, num)
        else:
            self.graph.plot(data)
        self.graph.getPlotItem().getViewBox().setMouseEnabled(True, False)
        self.graph.getPlotItem().setLabel("left", y_label)
        self.graph.getPlotItem().setLabel("bottom", x_label)
        # self.graph.axes.xticks(ticks=list(range(data.shape[0])), labels=x_ticks)
        # if display_y_axis_ticks and y_ticks != None:
        #     self.graph.axes.yaxis.set_ticks(ticks=list(range(data.shape[0])))
        #     self.graph.axes.set_yticklabels(labels=y_ticks)
        # if not display_y_axis_ticks:
        #     self.graph.axes.yaxis.set_ticks([])

        # layout_h.addWidget(QLabel(y_label))
        # test = QLabel("")
        # test.setMinimumHeight(200)
        # layout_h.addWidget(test)
        layout_h.addWidget(self.graph)

        layout.addLayout(layout_h, stretch=10)
        # layout.addWidget(QLabel(x_label), alignment=QtCore.Qt.AlignCenter,stretch=1)

        self.setLayout(layout)
        # self.setMinimumHeight(200)
        layout.setContentsMargins(5, 5, 5, 5)
        layout_h.setContentsMargins(0, 0, 00, 0)
