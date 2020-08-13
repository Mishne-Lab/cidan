import os

import requests
from qtpy import QtWidgets


def downloadDemo(save_path):
    url = ""
    if not os.path.isdir(save_path) and save_path != "":
        error_dialog = QtWidgets.QErrorMessage()
        error_dialog.showMessage("Please Select a valid folder")
    demo_folder_path = os.path.join(save_path, "CIDAN_Demo/")
    os.mkdir(demo_folder_path)
    r = requests.get(url)
    demo_dataset_path = os.path.join(demo_folder_path, "demo_dataset.tif")
    with open(demo_dataset_path, 'wb') as f:
        f.write(r.content)
