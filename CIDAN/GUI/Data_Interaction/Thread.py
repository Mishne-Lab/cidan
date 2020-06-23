from CIDAN.GUI.Data_Interaction.Signals import *


class Thread(QThread):
    def __init__(self, main_widget, parent=None):
        QThread.__init__(self, parent)
        self.exiting = False
        self.main_widget = main_widget
        self.signal = BoolSignal()

    @property
    def data_handler(self):
        return self.main_widget.data_handler
    def run(self):
        pass

    def endThread(self, success):
        pass
