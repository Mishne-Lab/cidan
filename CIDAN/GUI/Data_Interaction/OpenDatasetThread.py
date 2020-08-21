import logging
import sys

from qtpy import QtWidgets

from CIDAN.GUI.Data_Interaction.DataHandler import DataHandler
from CIDAN.GUI.Data_Interaction.Signals import StrIntSignal
from CIDAN.GUI.Data_Interaction.Thread import Thread

logger = logging.getLogger("CIDAN.GUI.Data_Interaction.PreprocessThread")


class OpenDatasetThread(Thread):

    def __init__(self, main_widget):
        super().__init__(main_widget)
        self.reportProgress = StrIntSignal()
        self.reportProgress.sig.connect(main_widget.console.updateProgressBar)
        self.signal.sig.connect(lambda x: self.endThread(x))

    def run(self):
        if self.main_widget.dev:
            self.main_widget.data_handler = DataHandler(data_path=self.data_path,
                                                        trials=self.trials,
                                                        save_dir_path=self.save_dir_path,
                                                        save_dir_already_created=self.save_dir_already_created,
                                                        load_into_mem=self.load_into_mem)
            self.main_widget.data_handler.calculate_filters(self.reportProgress)

            self.signal.sig.emit(True)
        else:
            try:

                self.signal.sig.emit(True)
            except Exception as e:
                logger.error(e)
                print("Unexpected error:", sys.exc_info()[0])
                self.main_widget.console.updateText("Unexpected error: " +
                                                    sys.exc_info()[0])
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage("Unexpected error: " + str(e))
                self.signal.sig.emit(False)

    def runThread(self, data_path, trials, save_dir_path, save_dir_already_created,
                  load_into_mem):
        self.data_path = data_path
        self.trials = trials
        self.save_dir_path = save_dir_path
        self.save_dir_already_created = save_dir_already_created
        self.load_into_mem = load_into_mem
        if not any([x.isRunning() for x in self.main_widget.thread_list]):
            print("Opening Dataset")
            self.main_widget.console.updateText("Opening Dataset")
            # self.button.setEnabled(False)
            self.start()
        else:
            print(
                "Previous process in process, please wait to start new one till "
                "finished")

    def endThread(self, success):
        self.main_widget.init_w_data()
