from CIDAN.GUI.Data_Interaction.Signals import *
from CIDAN.GUI.Data_Interaction.Thread import Thread


class ROIExtractionThread(Thread):
    def __init__(self, main_widget, button, roi_list_module, roi_tab):
        super().__init__(main_widget.data_handler)
        self.roi_tab = roi_tab
        self.main_widget = main_widget
        self.roi_list_module = roi_list_module
        self.button = button
        self.signal.sig.connect(lambda x: self.endThread(x))

    def run(self):
        try:
            self.data_handler.calculate_roi_extraction()
            self.signal.sig.emit(True)
        except AssertionError as e:
            self.signal.sig.emit(False)

    def runThread(self):

        if not any([x.isRunning() for x in self.main_widget.thread_list]):
            print("Starting ROI extraction")
            self.button.setEnabled(False)
            self.start()
        else:
            print(
                "Previous process in process, please wait to start new one till finished")

    def endThread(self, success):
        self.button.setEnabled(True)
        if success:
            print("Finished ROI extraction")
            self.roi_list_module.set_list_items(self.main_widget.data_handler.clusters)
            shape = self.main_widget.data_handler.edge_roi_image_flat.shape
            self.roi_tab.roi_image_flat = np.hstack(
                [self.main_widget.data_handler.edge_roi_image_flat,
                 np.zeros(shape),
                 np.zeros(shape)])
            self.roi_tab.select_image_flat = np.zeros([shape[0], 3])

            self.roi_tab.updateImageDisplay(new=True)
