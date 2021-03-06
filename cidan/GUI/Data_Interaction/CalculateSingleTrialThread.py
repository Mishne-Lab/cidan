import logging
import sys

from qtpy import QtWidgets

from cidan.GUI.Data_Interaction.Signals import StrIntSignal
from cidan.GUI.Data_Interaction.Thread import Thread

logger = logging.getLogger("cidan.GUI.Data_Interaction.CalculateSingleTrialThread")


class CalculateSingleTrialThread(Thread):

    def __init__(self, main_widget):
        super().__init__(main_widget)
        self.trial_num = 0
        self.end_function = min
        self.signal.sig.connect(lambda x: self.endThread(x))

    def run(self):
        if self.main_widget.dev:

            self.data_handler.dataset_trials_filtered_loaded[self.trial_num].compute()
            self.signal.sig.emit(True)
        else:
            try:
                self.data_handler.dataset_trials_filtered_loaded[self.trial_num].compute()
            except Exception as e:
                logger.error(e)
                print("Unexpected error:", sys.exc_info()[0])
                self.main_widget.console.updateText("Unexpected error: " +
                                                    sys.exc_info()[0])
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage("Unexpected error: " + str(e))
                self.signal.sig.emit(False)

    def runThread(self, trial_num, function):

        if not any([x.isRunning() for x in self.main_widget.thread_list]):
            self.trial_num = trial_num
            self.end_function = function
            self.start()
        else:
            print(
                "Previous process in process, please wait to start new one till "
                "finished")

    def endThread(self, success):
        self.end_function()

