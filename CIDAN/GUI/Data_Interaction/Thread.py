from CIDAN.GUI.Data_Interaction.Signals import *


class Thread(QThread):
    def __init__(self, data_handler, parent=None):
        QThread.__init__(self, parent)
        self.exiting = False
        self.signal = BoolSignal()
        self.data_handler = data_handler

    def run(self):
        pass

    def endThread(self, success):
        pass
