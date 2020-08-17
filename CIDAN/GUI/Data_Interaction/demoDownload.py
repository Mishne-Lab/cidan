import os

import requests
from qtpy import QtWidgets


def demoDownload(save_path):
    url = "https://github.com/Mishne-Lab/CIDAN/blob/master/demo/demo_dataset_1.tif?raw=true"
    if not os.path.isdir(save_path) and save_path != "":
        error_dialog = QtWidgets.QErrorMessage()
        error_dialog.showMessage("Please Select a valid folder")
    demo_folder_path = os.path.join(save_path, "CIDAN_Demo/")
    if not os.path.isdir(demo_folder_path):
        os.mkdir(demo_folder_path)
    r = requests.get(url)
    demo_dataset_path = os.path.join(demo_folder_path, "demo_dataset_1.tif")
    with open(demo_dataset_path, 'wb') as f:
        f.write(r.content)
    return True
