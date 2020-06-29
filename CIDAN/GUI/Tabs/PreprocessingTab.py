from qtpy import QtCore
from qtpy.QtWidgets import *

from CIDAN.GUI.Data_Interaction.PreprocessThread import PreprocessThread
from CIDAN.GUI.Inputs.OptionInput import OptionInput
from CIDAN.GUI.SettingWidget.SettingsModule import preprocessing_settings
from CIDAN.GUI.Tabs.Tab import Tab


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
        process_button_layout.setContentsMargins(2, 2, 2, 2)
        thread = PreprocessThread(main_widget, process_button, self)
        main_widget.thread_list.append(thread)  # Appends the thread to the main
        # widget thread list
        process_button.clicked.connect(lambda: thread.runThread())
        # This assumes that the data is already loaded in
        self.data_handler.calculate_filters()


        # Section that creates all the buttons to change which image is displayed
        image_buttons = QWidget()
        self._image_buttons_layout = QHBoxLayout()
        self._image_buttons_layout.setContentsMargins(2, 0, 2, 0)
        image_buttons.setLayout(self._image_buttons_layout)
        max_image_button = QPushButton()
        max_image_button.setText("Max Image")
        max_image_button.clicked.connect(
            lambda: self.set_image_display_list(self.data_handler.trials_loaded,
                                                self.data_handler.max_images))
        stack_button = QPushButton()
        stack_button.setText("Filtered Stack")
        stack_button.clicked.connect(
            lambda: self.set_image_display_list(self.data_handler.trials_loaded,
                                                self.data_handler.dataset_trials_filtered_loaded))
        self.pca_stack_button = QPushButton()
        self.pca_stack_button.setText("PCA Stack")
        self.pca_stack_button.clicked.connect(
            lambda: self.set_image_display_list(self.data_handler.trials_loaded,
                                                self.data_handler.pca_decomp))
        mean_image_button = QPushButton()
        mean_image_button.setText("Mean Image")
        mean_image_button.clicked.connect(
            lambda: self.set_image_display_list(self.data_handler.trials_loaded,
                                                self.data_handler.mean_images))

        self._image_buttons_layout.addWidget(stack_button)
        self._image_buttons_layout.addWidget(max_image_button)
        self._image_buttons_layout.addWidget(mean_image_button)
        self._image_buttons_layout.addWidget(self.pca_stack_button)

        main_widget.preprocess_image_view.setContentsMargins(0, 0, 0, 0)
        # main_widget.preprocess_image_view.setMargin(0)
        preprocessing_settings_widget = preprocessing_settings(main_widget)
        preprocessing_settings_widget.setContentsMargins(0, 0, 0, 0)
        # Update image view
        self.updateTab()
        # Initialize the tab with the necessary columns
        super().__init__("Preprocessing", column_1=[preprocessing_settings_widget
                                                    ],
                         column_2=[main_widget.preprocess_image_view
                                   ])
        self.column_1_layout.addWidget(process_button_widget,
                                       alignment=QtCore.Qt.AlignBottom)
        self.column_2_layout.addWidget(image_buttons, alignment=QtCore.Qt.AlignBottom)

    @property
    def data_handler(self):
        return self.main_widget.data_handler
    def set_image_display_list(self, trial_names, data_list):
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

        def set_image(x, trial_name):
            self.main_widget.preprocess_image_view.setImage(
                data_list[trial_names.index(trial_name)][:])

        if hasattr(self, "trial_selector_input"):
            self.trial_selector_input.setParent(None)
        self.trial_selector_input = OptionInput("", "", set_image, val_list=trial_names,
                                                tool_tip="Select Trial to display",
                                                display_tool_tip=False, default_index=0,
                                                show_name=False)
        self._image_buttons_layout.addWidget(self.trial_selector_input)
        set_image("", trial_names[0])

    def set_image_display(self, data):
        self.trial_selector_input.setParent(None)
        self.trial_selector_input = OptionInput("", "", lambda x, y: 3,
                                                val_list=["All"],
                                                tool_tip="Select Trial to display",
                                                display_tool_tip=False, default_index=0,
                                                show_name=False)
        self._image_buttons_layout.addWidget(self.trial_selector_input)
        self.main_widget.preprocess_image_view.setImage(
            data)

    def updateTab(self):
        if (self.main_widget.checkThreadRunning()):
            self.pca_stack_button.setEnabled(self.data_handler.filter_params["pca"])
            self.set_image_display_list(self.data_handler.trials_loaded,
                                        self.data_handler.max_images)
