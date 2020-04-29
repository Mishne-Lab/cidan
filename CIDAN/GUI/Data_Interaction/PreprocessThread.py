from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *
from CIDAN.GUI.Data_Interaction.Signals import *
from CIDAN.GUI.Data_Interaction.Thread import Thread


class PreprocessThread(Thread):
    def __init__(self, main_widget, button):
        super().__init__(main_widget.data_handler)
        self.main_widget = main_widget
        self.button = button
        self.signal.sig.connect(lambda x: self.endThread(x))

    def run(self):
        try:
            self.signal.sig.emit(self.data_handler.calculate_filters())
        except:
            print("Unexpected error:", sys.exc_info()[0])
            self.signal.sig.emit(np.matrix([0]))

    def runThread(self):

        if not any([x.isRunning() for x in self.main_widget.thread_list]):
            print("Starting preprocessing sequence")
            self.button.setEnabled(False)
            self.start()
        else:
            print(
                "Previous process in process, please wait to start new one till finished")

    def endThread(self, image_data):
        self.button.setEnabled(True)
        if image_data.shape != [1]:
            print("Finished preprocessing sequence")
            self.main_widget.preprocess_image_view.setImage(image_data)
