from CIDAN.GUI.Data_Interaction.Signals import *


class Worker(QRunnable):
    def __init__(self, main_widget, parent=None):
        super(Worker, self).__init__()
        self.main_widget = main_widget
        self.signal = BoolSignal()

    @property
    def data_handler(self):
        return self.main_widget.data_handler
    def run(self):
        pass

    def endThread(self, success):
        pass

    def __del__(self):
        self.exiting = True
        self.wait()
