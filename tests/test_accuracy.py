import shutil

import neurofinder
from PySide2.QtWidgets import QApplication
from skimage import io
from skimage.metrics import structural_similarity

from CIDAN.GUI.Data_Interaction.loadDataset import load_new_dataset
from CIDAN.GUI.MainWindow import MainWindow


def test_accuracy():
    app = QApplication([])
    app.setApplicationName("CIDAN")
    widget = MainWindow(dev=True, preload=False)

    main_widget = widget.table_widget
    load_new_dataset(main_widget, file_path="test_files/small_dataset1.tif",
                     save_dir_path="test_files/save_dir", load_into_mem=True)

    data_handler = main_widget.data_handler

    data_handler.change_filter_param("median_filter", True)
    data_handler.change_roi_extraction_param("roi_circ_threshold", 40)
    main_widget.thread_list[1].run()
    assert main_widget.tabs[1].image_view.magic_wand(70, 127)

    data_handler.export()
    a = neurofinder.load("test_files/roi_list.json")
    b = neurofinder.load("test_files/save_dir/roi_list.json")
    test = neurofinder.match(a, b)
    percision, recall = neurofinder.centers(a, b)
    inclusion, exclusion = neurofinder.shapes(a, b)
    assert percision > .8
    assert recall > .8
    assert exclusion > .8
    assert inclusion > .8
    image_good = io.imread("test_files/embedding_norm_image.png")
    image_test = io.imread(
        "test_files/save_dir/embedding_norm_images/embedding_norm_image.png")

    assert structural_similarity(image_good, image_test) > .90
    data_handler.change_box_param("total_num_spatial_boxes", 4)
    main_widget.thread_list[1].run()
    data_handler.export()
    image_test = io.imread(
        "test_files/save_dir/embedding_norm_images/embedding_norm_image.png")

    assert structural_similarity(image_good, image_test) > .60
    a = neurofinder.load("test_files/roi_list.json")
    b = neurofinder.load("test_files/save_dir/roi_list.json")
    percision, recall = neurofinder.centers(a, b)
    inclusion, exclusion = neurofinder.shapes(a, b)
    assert percision > .4
    assert recall > .4
    assert exclusion > .4
    assert inclusion > .3
    app.quit()
    shutil.rmtree("test_files/save_dir")
