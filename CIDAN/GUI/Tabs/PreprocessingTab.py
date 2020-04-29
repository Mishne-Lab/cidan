from CIDAN.GUI.Tabs.Tab import Tab
from PySide2.QtWidgets import *
from CIDAN.GUI.Data_Interaction.PreprocessThread import PreprocessThread
from CIDAN.GUI.SettingWidget.SettingsModule import preprocessing_settings


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
        self.data_handler = self.main_widget.data_handler

        # This part initializes the button to process the data
        process_button = QPushButton()
        process_button.setText("Apply Settings")
        thread = PreprocessThread(main_widget, process_button)
        main_widget.thread_list.append(thread) # Appends the thread to the main
        # widget thread list
        process_button.clicked.connect(lambda: thread.runThread())
        # This assumes that the data is already loaded in
        self.main_widget.preprocess_image_view.setImage(
            self.data_handler.calculate_filters())

        # Section that creates all the buttons to change which image is displayed
        image_buttons = QWidget()
        image_buttons_layout = QHBoxLayout()
        image_buttons.setLayout(image_buttons_layout)
        max_image_button = QPushButton()
        max_image_button.setText("Max Image")
        max_image_button.clicked.connect(
            lambda: self.main_widget.preprocess_image_view.setImage(
                self.data_handler.max_image))
        stack_button = QPushButton()
        stack_button.setText("Filtered Stack")
        stack_button.clicked.connect(
            lambda: self.main_widget.preprocess_image_view.setImage(
                self.data_handler.dataset_filtered))
        orig_stack_button = QPushButton()
        orig_stack_button.setText("Original Stack")
        orig_stack_button.clicked.connect(
            lambda: self.main_widget.preprocess_image_view.setImage(
                self.data_handler.dataset))
        mean_image_button = QPushButton()
        mean_image_button.setText("Mean Image")
        mean_image_button.clicked.connect(
            lambda: self.main_widget.preprocess_image_view.setImage(
                self.data_handler.mean_image))
        image_buttons_layout.addWidget(orig_stack_button)
        image_buttons_layout.addWidget(stack_button)
        image_buttons_layout.addWidget(max_image_button)
        image_buttons_layout.addWidget(mean_image_button)

        # Initialize the tab with the necessary columns
        super().__init__("Preprocessing", column_1=[preprocessing_settings(main_widget),
                                                    process_button], column_2=[
            main_widget.preprocess_image_view, image_buttons])
