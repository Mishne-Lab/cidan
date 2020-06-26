from qtpy.QtWidgets import *

from CIDAN.GUI.Data_Interaction.loadDataset import *
from CIDAN.GUI.Inputs.BoolInput import BoolInput
from CIDAN.GUI.Inputs.FileInput import FileInput
from CIDAN.GUI.ListWidgets.TrialListWidget import TrialListWidget
from CIDAN.GUI.Tabs.Tab import Tab


class FileOpenTab(Tab):
    def __init__(self, main_widget):
        # TODO Make this less ugly can reorganize code
        self.trial_list_widget = TrialListWidget()
        dataset_file_input = FileInput("Dataset File:", "", None, "",
                                       "Select a file to load in", isFolder=2,
                                       forOpen=True)
        dataset_folder_input = FileInput("Dataset Folder:", "",
                                         lambda x: self.add_trial_selection(x), "",
                                         "Select a folder to load in", isFolder=1,
                                         forOpen=True)
        save_dir_new_file = FileInput("Save Directory Location:", "", None, "",
                                      "Select a place to save outputs", isFolder=1,
                                      forOpen=False)
        save_dir_new_folder = FileInput("Save Directory Location:", "", None, "",
                                        "Select a place to save outputs", isFolder=1,
                                        forOpen=False)

        save_dir_load = FileInput("Previous Session Location:", "", None, "",
                                  tool_tip="Select the save directory for a previous session",
                                  isFolder=1, forOpen=True)

        file_open_button = QPushButton()
        file_open_button.setContentsMargins(0, 0, 0, 11)
        file_open_button.setText("Load")
        file_open_button.clicked.connect(
            lambda: load_new_dataset(main_widget, dataset_file_input.current_state(),
                                     save_dir_new_file.current_state(),
                                     load_into_mem=self.folder_load_into_mem.current_state()))
        self.file_load_into_mem = BoolInput("Load data into memory(not recomended "
                                            "for large datasets)", "", None, False,
                                            "")
        folder_open_button = QPushButton()
        folder_open_button.setContentsMargins(0, 0, 0, 11)
        folder_open_button.setText("Load")
        folder_open_button.clicked.connect(
            lambda: load_new_dataset(main_widget, dataset_folder_input.current_state(),
                                     save_dir_new_folder.current_state(),
                                     trials=self.trial_list_widget.selectedTrials(),
                                     single=self.folder_open_single_trial.current_state(),
                                     load_into_mem=self.folder_load_into_mem.current_state()))
        self.folder_open_single_trial = BoolInput("Open folder as single trial", "",
                                                  None, False, "")
        self.folder_load_into_mem = BoolInput("Load data into memory(not recomended "
                                              "for large datasets)", "", None, False,
                                              "")
        prev_session_open_button = QPushButton()
        prev_session_open_button.setContentsMargins(0, 0, 0, 11)
        prev_session_open_button.setText("Load")
        prev_session_open_button.clicked.connect(
            lambda: load_prev_session(main_widget, save_dir_load.current_state()))
        file_open = Tab("File Open", column_2=[], column_2_display=False,
                        column_1=[dataset_file_input, save_dir_new_file,
                                  self.file_load_into_mem,
                                  file_open_button])
        folder_open = Tab("Folder Open", column_2=[self.trial_list_widget],
                          column_2_display=True,
                          column_1=[dataset_folder_input, save_dir_new_folder,
                                    self.folder_open_single_trial,
                                    self.folder_load_into_mem,
                                    folder_open_button]
                          )
        prev_session_open = Tab("Previous Session Open", column_2=[],
                                column_2_display=False,
                                column_1=[save_dir_load, prev_session_open_button])
        self.tab_selector = QTabWidget()

        self.tab_selector.addTab(file_open, file_open.name)
        self.tab_selector.addTab(folder_open, folder_open.name)
        self.tab_selector.addTab(prev_session_open, prev_session_open.name)

        super().__init__("FileOpenTab", column_1=[self.tab_selector], column_2=[],
                         column_2_display=False)

    def add_trial_selection(self, path):
        self.trial_list_widget.set_items_from_path(path)
