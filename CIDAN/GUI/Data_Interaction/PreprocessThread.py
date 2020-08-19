import logging
import sys

from qtpy import QtWidgets

from CIDAN.GUI.Data_Interaction.Signals import StrIntSignal
from CIDAN.GUI.Data_Interaction.Thread import Thread

logger = logging.getLogger("CIDAN.GUI.Data_Interaction.PreprocessThread")


class PreprocessThread(Thread):

    def __init__(self, main_widget, button, preprocess_tab):
        super().__init__(main_widget)
        self.button = button
        self.preprocess_tab = preprocess_tab
        self.reportProgress = StrIntSignal()
        self.reportProgress.sig.connect(main_widget.console.updateProgressBar)
        self.signal.sig.connect(lambda x: self.endThread(x))

    def run(self):
        if self.main_widget.dev:

            self.data_handler.calculate_filters(self.reportProgress)
            self.signal.sig.emit(True)
        else:
            try:
                self.data_handler.calculate_filters(self.reportProgress)
                self.signal.sig.emit(True)
            except Exception as e:
                logger.error(e)
                print("Unexpected error:", sys.exc_info()[0])
                self.main_widget.console.updateText("Unexpected error: " +
                                                    sys.exc_info()[0])
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage("Unexpected error: " + str(e))
                self.signal.sig.emit(False)

    def runThread(self):

        if not any([x.isRunning() for x in self.main_widget.thread_list]):
            print("Starting preprocessing sequence")
            self.main_widget.console.updateText("Starting preprocessing sequence")
            # self.button.setEnabled(False)
            self.start()
        else:
            print(
                "Previous process in process, please wait to start new one till "
                "finished")

    def endThread(self, success):
        self.button.setEnabled(True)
        if success:
            print("Finished preprocessing sequence")

            self.main_widget.updateTabs()
            self.main_widget.console.updateText("Finished preprocessing sequence")
