import dask
from qtpy import QtCore
from qtpy.QtWidgets import *

from cidan.GUI.Data_Interaction.PreprocessThread import PreprocessThread
from cidan.GUI.Inputs.OptionInput import OptionInput
from cidan.GUI.SettingWidget.SettingsModule import preprocessing_settings
from cidan.GUI.Tabs.Tab import Tab


class PreprocessingTab(Tab):
    """Class controlling the Preprocessing tab, inherits from Tab


    Attributes
    ----------
    main_widget : MainWidget
        A reference to the main widget
    data_handler : DataHandler
        A reference to the main DataHandler of MainWidget

    """

    def __init__(self, main_widget):
        self.main_widget = main_widget
        self.image_view = self.main_widget.preprocess_image_view
        # This part initializes the button to process the data
        process_button = QPushButton()
        process_button.setText("Apply Settings")
        process_button_layout = QVBoxLayout()
        process_button_widget = QWidget()
        process_button_widget.setLayout(process_button_layout)
        process_button_layout.addWidget(process_button)
        process_button_layout.setContentsMargins(0, 0, 0, 0)
        thread = PreprocessThread(main_widget, process_button, self)
        main_widget.thread_list.append(thread)  # Appends the thread to the main
        # widget thread list
        process_button.clicked.connect(lambda: thread.runThread())

        # Section that creates all the buttons to change which image is displayed
        image_buttons = QWidget()
        self._image_buttons_layout = QHBoxLayout()
        self._image_buttons_layout.setContentsMargins(2, 0, 2, 0)
        image_buttons.setLayout(self._image_buttons_layout)
        self.max_image_button = QPushButton()
        self.max_image_button.setText("Max Image")
        self.max_image_button.clicked.connect(
            lambda: self.set_image_display_list(self.data_handler.trials_loaded,
                                                self.data_handler.max_images,
                                                "Max Image", self.max_image_button))
        stack_button = QPushButton()
        stack_button.setText("Filtered Stack")
        stack_button.clicked.connect(
            lambda: self.set_image_display_list(self.data_handler.trials_loaded,
                                                self.data_handler.dataset_trials_filtered_loaded,
                                                "Filtered Stack", stack_button))
        self.pca_stack_button = QPushButton()
        self.pca_stack_button.setText("PCA Stack")
        self.pca_stack_button.clicked.connect(
            lambda: self.set_image_display_list(self.data_handler.trials_loaded,
                                                self.data_handler.pca_decomp,
                                                "PCA Stack", self.pca_stack_button))
        self.mean_image_button = QPushButton()
        self.mean_image_button.setText("Mean Image")
        self.mean_image_button.clicked.connect(
            lambda: self.set_image_display_list(self.data_handler.trials_loaded,
                                                self.data_handler.mean_images,
                                                "Mean Image", self.mean_image_button))
        self.temporal_correlation_image_button = QPushButton()
        self.temporal_correlation_image_button.setText("Temporal Correlation Image")
        self.temporal_correlation_image_button.clicked.connect(
            lambda: self.set_image_display_list(self.data_handler.trials_loaded,
                                                self.data_handler.temporal_correlation_images,
                                                "Temporal Correlation Image",
                                                self.temporal_correlation_image_button))
        self._image_buttons_layout.addWidget(stack_button)
        self._image_buttons_layout.addWidget(self.max_image_button)
        self._image_buttons_layout.addWidget(self.mean_image_button)
        # self._image_buttons_layout.addWidget(self.temporal_correlation_image_button)
        self._image_buttons_layout.addWidget(self.pca_stack_button)

        main_widget.preprocess_image_view.setContentsMargins(0, 0, 0, 0)
        # main_widget.preprocess_image_view.setMargin(0)
        preprocessing_settings_widget = preprocessing_settings(main_widget)
        preprocessing_settings_widget.setContentsMargins(0, 0, 0, 0)
        # preprocessing_settings_widget.setMaximumWidth(400)
        # preprocessing_settings_widget.setMinimumWidth(int(400*((self.logicalDpiX() / 96.0-1)/2+1)))
        # main_widget.preprocess_image_view.setMinimumWidth(500)
        # Update image view
        self.updateTab()
        main_widget.preprocess_image_view.setSizePolicy(QSizePolicy.Expanding,
                                                        QSizePolicy.Expanding)
        # Initialize the tab with the necessary columns
        super().__init__("Preprocessing", column_1=[preprocessing_settings_widget
                                                    ],
                         column_2=[main_widget.preprocess_image_view
                                   ], horiz_moveable=True)
        # process_button_widget.setMaximumWidth(400*((self.logicalDpiX() / 96.0-1)/2+1)

        self.layout.setContentsMargins(0, 0, 0, 0)
        self.column_1_layout.addWidget(process_button_widget,
                                       alignment=QtCore.Qt.AlignBottom)
        self.column_1_layout.setContentsMargins(0, 0, 0, 0)
        self.column_2_layout.addWidget(image_buttons, alignment=QtCore.Qt.AlignBottom)
        self.column_2_layout.setContentsMargins(0, 0, 0, 0)

    @property
    def data_handler(self):
        return self.main_widget.data_handler

    def set_image_display_list(self, trial_names, data_list, name, button=None):
        """
        Sets the preprocessing image display to use an option input and set data list
        Parameters
        ----------
        trial_names : List[str]
            the names of each trial
        data_list : List[np.ndarray]
            Corresponding data for each trial
        Returns
        -------
        Nothing
        """
        if hasattr(self.data_handler, "total_size"):
            total_size = (
                str(self.data_handler.total_size[0]),
                str(self.data_handler.total_size[1]))
        else:
            total_size = [0, 0]
        if len(data_list[0].shape) == 3:
            cur_size = [str(data_list[0].shape[1]), str(data_list[0].shape[2])]
        else:
            cur_size = [str(data_list[0].shape[0]), str(data_list[0].shape[1])]
        def set_image(x, trial_name):
            def end_func():
                try:
                    self.main_widget.preprocess_image_view.setImage(
                        data_list[trial_names.index(trial_name)][:])
                    self.main_widget.console.updateText(
                        "Loaded filtered trial %s" % str(trial_name))
                    self.image_view.image_label.setText(
                        "Trial: %s, " % str(
                            trial_name) + name + ", Original Size: (%s, %s), Cropped Size: (%s, %s)" % (
                            total_size[1], total_size[0], cur_size[1], cur_size[0]))
                    self.current_trial_name = trial_name
                except TypeError:
                    self.updateTab()

            if (self.main_widget.checkThreadRunning()):
                try:
                    if type(data_list[trial_names.index(trial_name)]) == bool or \
                            type(self.data_handler.dataset_trials_filtered_loaded[
                                     trial_names.index(trial_name)]) == type(
                        dask.delayed(min)()):
                        self.main_widget.console.updateText(
                            "Applying filters to trial %s" % str(trial_name))
                        self.main_widget.calculate_single_trial_thread.runThread(
                            trial_names.index(trial_name), end_func)
                    else:
                        self.main_widget.preprocess_image_view.setImage(
                            data_list[trial_names.index(trial_name)][:])
                        self.main_widget.console.updateText(
                            "Loaded filtered trial %s" % str(trial_name))
                        self.image_view.image_label.setText(
                            "Trial: %s, " % str(
                                trial_name) + name + ", Original Size: (%s, %s), Cropped Size: (%s, %s)" % (
                                total_size[1], total_size[0], cur_size[1], cur_size[0]))
                        self.current_trial_name = trial_name
                except TypeError:
                    self.updateTab()
            elif self.trial_selector_input.current_state() != self.current_trial_name:
                self.trial_selector_input.set_val(self.current_trial_name)
                print("Please wait until current process is done")

        if (self.main_widget.checkThreadRunning()):
            if hasattr(self, "trial_selector_input"):
                self.trial_selector_input.setParent(None)
            if hasattr(self, "trial_selector_input"):
                current = self.trial_selector_input.input_box.currentIndex()
            else:
                current = 0
            # if button != None:
            #     button.setStyleSheet("QPushButton {border 1px solid #148CD2; }")
            self.trial_selector_input = OptionInput("", "", set_image,
                                                    val_list=trial_names,
                                                    tool_tip="Select time block to display",
                                                    display_tool_tip=False,
                                                    default_index=0,
                                                    show_name=False)
            self.trial_selector_input.input_box.setCurrentIndex(current)
            self._image_buttons_layout.addWidget(self.trial_selector_input)
            self.current_trial_name = trial_names[current]
            set_image("", trial_names[current])

            self.main_widget.console.updateText(
                "Setting current background to: %s" % name)
        else:
            print("Please wait until current process is done")
    def set_image_display(self, data):
        self.trial_selector_input.setParent(None)
        self.trial_selector_input = OptionInput("", "", lambda x, y: 3,
                                                val_list=["All"],
                                                tool_tip="Select Timeblock to display",
                                                display_tool_tip=False, default_index=0,
                                                show_name=False)
        self._image_buttons_layout.addWidget(self.trial_selector_input)
        self.main_widget.preprocess_image_view.setImage(
            data.copy())

    def updateTab(self):
        if (self.main_widget.checkThreadRunning()):
            self.pca_stack_button.setEnabled(self.data_handler.filter_params["pca"])
            self.set_image_display_list(self.data_handler.trials_loaded,
                                        self.data_handler.max_images, "Max Image")
