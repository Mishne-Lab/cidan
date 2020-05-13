import sys

from CIDAN.GUI.Data_Interaction.Thread import Thread


class PreprocessThread(Thread):
    def __init__(self, main_widget, button, preprocess_tab):
        super().__init__(main_widget.data_handler)
        self.main_widget = main_widget
        self.button = button
        self.preprocess_tab = preprocess_tab
        self.signal.sig.connect(lambda x: self.endThread(x))

    def run(self):
        try:
            self.data_handler.calculate_filters()
            self.signal.sig.emit(True)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            self.signal.sig.emit(False)

    def runThread(self):

        if not any([x.isRunning() for x in self.main_widget.thread_list]):
            print("Starting preprocessing sequence")
            self.button.setEnabled(False)
            self.start()
        else:
            print(
                "Previous process in process, please wait to start new one till "
                "finished")

    def endThread(self, success):
        self.button.setEnabled(True)
        if success:
            print("Finished preprocessing sequence")
            self.preprocess_tab.set_image_display_list(self.data_handler.trials_loaded,
                                                       self.data_handler.dataset_trials_filtered_loaded)
