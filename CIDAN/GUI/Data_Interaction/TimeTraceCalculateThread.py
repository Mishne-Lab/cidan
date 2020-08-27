import logging
import sys

from qtpy import QtWidgets

from CIDAN.GUI.Data_Interaction.Signals import StrIntSignal
from CIDAN.GUI.Data_Interaction.Thread import Thread

logger = logging.getLogger("CIDAN.GUI.Data_Interaction.PreprocessThread")


class TimeTraceCalculateThread(Thread):

    def __init__(self, main_widget, button, preprocess_tab):
        super().__init__(main_widget)
        self.button = button
        self.preprocess_tab = preprocess_tab
        self.reportProgress = StrIntSignal()
        self.reportProgress.sig.connect(main_widget.console.updateProgressBar)
        self.signal.sig.connect(lambda x: self.endThread(x))

    def run(self):

        if self.main_widget.dev:

            self.data_handler.calculate_time_traces(self.reportProgress)
            self.signal.sig.emit(True)
        else:
            try:
                self.data_handler.calculate_time_traces(self.reportProgress)
                self.signal.sig.emit(True)
            except Exception as e:
                ex_info = sys.exc_info()[0]
                logger.error(e)
                logger.error(ex_info)
                print("Unexpected error:", sys.exc_info()[0])
                self.main_widget.console.updateText("Unexpected error: " +
                                                    ex_info)
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage("Unexpected error: " + str(e))
                self.signal.sig.emit(False)

    def runThread(self, end_func=lambda: 2):
        self.end_func = end_func

        if not any([x.isRunning() for x in self.main_widget.thread_list]):
            print("Starting calculating time traces")
            self.main_widget.console.updateText("Starting calculating time traces")
            # self.button.setEnabled(False)
            self.start()
        else:
            print(
                "Previous process in process, please wait to start new one till "
                "finished")

    def endThread(self, success):
        self.button.setEnabled(True)
        if success:
            print("Finished calculating time traces")

            self.end_func()
            self.main_widget.console.updateText("Finished calculating time traces")
