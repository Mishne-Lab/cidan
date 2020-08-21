import logging
import sys

from qtpy import QtWidgets

from CIDAN.GUI.Data_Interaction.Signals import StrIntSignal
from CIDAN.GUI.Data_Interaction.Thread import Thread
from CIDAN.GUI.Data_Interaction.demoDownload import demoDownload

logger = logging.getLogger("CIDAN.GUI.Data_Interaction.PreprocessThread")


class DemoDownloadThread(Thread):

    def __init__(self, main_widget):
        super().__init__(main_widget)
        self.reportProgress = StrIntSignal()
        self.reportProgress.sig.connect(main_widget.console.updateProgressBar)
        self.signal.sig.connect(lambda x: self.endThread(x))

    def run(self):
        if self.main_widget.dev:

            self.signal.sig.emit(demoDownload(self.path))
        else:
            try:

                self.signal.sig.emit(demoDownload(self.path))
            except Exception as e:
                logger.error(e)
                print("Unexpected error:", sys.exc_info()[0])
                self.main_widget.console.updateText("Unexpected error: " +
                                                    sys.exc_info()[0])
                error_dialog = QtWidgets.QErrorMessage()
                error_dialog.showMessage("Unexpected error: " + str(e))
                self.signal.sig.emit(False)

    def runThread(self, path, endfunc):
        self.path = path
        self.end_func = endfunc
        if not any([x.isRunning() for x in self.main_widget.thread_list]):
            print("Downloading Demo Dataset")
            self.main_widget.console.updateText("Downloading Demo Dataset")
            # self.button.setEnabled(False)
            self.start()
        else:
            print(
                "Previous process in process, please wait to start new one till "
                "finished")

    def endThread(self, success):
        self.end_func(success)
