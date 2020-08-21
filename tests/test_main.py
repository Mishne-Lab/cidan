import shutil
import time

from PySide2.QtWidgets import QApplication

from CIDAN.GUI.Data_Interaction.loadDataset import load_new_dataset, load_prev_session
from CIDAN.GUI.MainWindow import MainWindow


def test_main():
    try:
        app = QApplication([])
        app.setApplicationName("CIDAN")
    except RuntimeError:
        pass
    widget = MainWindow(dev=True, preload=False)

    main_widget = widget.table_widget
    load_new_dataset(main_widget, file_path="test_files/small_dataset1.tif",
                     save_dir_path="test_files/save_dir", load_into_mem=False)
    main_widget.open_dataset_thread.wait()
    time.sleep(25)
    assert main_widget.data_handler.shape == [400, 150]
    assert main_widget.data_handler.rois_loaded == False
    assert main_widget.data_handler.trials_loaded == ["0"]
    assert main_widget.data_handler.trials_all == ["0"]
    assert main_widget.data_handler.trials_loaded_time_trace_indices == [0]
    assert main_widget.data_handler._trials_loaded_indices == [0]
    for tab in main_widget.tabs:
        assert tab.image_view.image_item.image.shape[0] == 400
        assert tab.image_view.image_item.image.shape[1] == 150
    data_handler = main_widget.data_handler
    data_handler.change_dataset_param("crop_x", [0, 200])
    data_handler.change_dataset_param("crop_stack", True)
    data_handler.change_filter_param("hist_eq", True)
    main_widget.thread_list[0].run()
    assert main_widget.data_handler.shape == [200, 150]

    data_handler.change_filter_param("hist_eq", False)
    data_handler.change_filter_param("median_filter", True)
    data_handler.change_roi_extraction_param("roi_circ_threshold", 0)
    main_widget.thread_list[1].run()
    assert main_widget.data_handler.shape == [200, 150]
    assert len(main_widget.data_handler.rois) != 0
    assert main_widget.data_handler.rois_loaded == True
    roi_image_view = main_widget.tabs[1].image_view
    roi_image_view.zoomRoi(2)
    roi_2_pixels = main_widget.data_handler.rois[1]
    for x in range(14):
        roi_image_view.pixel_paint(x, 2)
    main_widget.tabs[1].roi_list_module.set_current_select(2)
    assert main_widget.tabs[1].modify_roi(2, add_subtract="add", override=True)
    assert list(roi_2_pixels) != list(main_widget.data_handler.rois[1])
    assert not main_widget.tabs[1].modify_roi(2, add_subtract="add", override=True)

    for x in range(14):
        roi_image_view.pixel_paint(x, 2)
    main_widget.tabs[1].roi_list_module.set_current_select(2)
    assert main_widget.tabs[1].modify_roi(2, add_subtract="subtract", override=True)
    assert list(roi_2_pixels) == list(main_widget.data_handler.rois[1])
    num_rois = len(data_handler.rois)
    main_widget.tabs[1].delete_roi(2)
    assert num_rois - 1 == len(data_handler.rois)
    assert list(roi_2_pixels) != list(main_widget.data_handler.rois[1])
    roi_image_view.setBrushSize(5)
    for x in range(14):
        roi_image_view.pixel_paint(x, 2)
    main_widget.tabs[1].add_new_roi()
    assert num_rois == len(data_handler.rois)
    for x in range(14):
        roi_image_view.pixel_paint(x, 2)
    roi_image_view.select_mode = "subtract"
    for x in range(14):
        roi_image_view.pixel_paint(x, 2)
    assert not main_widget.tabs[1].modify_roi(2, add_subtract="add", override=True)
    # assert roi_image_view.magic_wand(50, 120)

    roi_image_view.set_background("", "Mean Image")
    roi_image_view.set_background("", "Blank Image")
    roi_image_view.set_background("", "Eigen Norm Image")
    roi_image_view.set_image("", "Blob")
    main_widget.tabs[2].roi_list_module.set_current_select(2)
    main_widget.tabs[2].roi_list_module.set_current_select(1)
    main_widget.tabs[2].plot_type_input.input_box.setCurrentIndex(1)
    main_widget.tabs[2].plot_by_input.input_box.setCurrentIndex(1)

    main_widget.tabs[2].roi_list_module.set_current_select(1)
    main_widget.tabs[2].plot_type_input.input_box.setCurrentIndex(0)
    main_widget.tabs[2].time_trace_type.input_box.setCurrentIndex(1)
    main_widget.tabs[2]._time_trace_trial_select_list.selectAll()
    main_widget.tabs[2].update_time_traces()
    main_widget.tabs[2].time_trace_type.input_box.setCurrentIndex(0)
    main_widget.tabs[2].roi_list_module.set_current_select(2)
    main_widget.tabs[2].roi_list_module.set_current_select(1)
    main_widget.tabs[2].plot_type_input.input_box.setCurrentIndex(1)
    main_widget.tabs[2].plot_by_input.input_box.setCurrentIndex(1)

    main_widget.tabs[2].roi_list_module.set_current_select(1)
    main_widget.tabs[2].plot_type_input.input_box.setCurrentIndex(0)
    main_widget.tabs[2].time_trace_type.input_box.setCurrentIndex(1)
    data_handler.change_filter_param("pca", True)
    main_widget.thread_list[0].run()
    print("test")
    main_widget.thread_list[1].run()
    assert main_widget.data_handler.shape == [200, 150]
    assert len(main_widget.data_handler.rois) != 0
    assert main_widget.data_handler.rois_loaded == True
    roi_image_view = None
    main_widget = None
    data_handler = None
    widget = None

    widget = MainWindow(dev=False, preload=False)
    main_widget = widget.table_widget
    load_prev_session(main_widget,
                      save_dir_path="test_files/save_dir")
    main_widget.open_dataset_thread.wait()
    assert len(main_widget.data_handler.rois) != 0
    assert main_widget.data_handler.shape == [200, 150]
    assert main_widget.data_handler.rois_loaded == True
    data_handler = main_widget.data_handler
    data_handler.change_dataset_param("crop_x", [0, 200])
    data_handler.change_dataset_param("crop_stack", True)
    data_handler.change_filter_param("hist_eq", True)
    main_widget.thread_list[0].run()
    assert data_handler.rois_loaded == False
    load_new_dataset(main_widget, file_path="test_files/",
                     save_dir_path="test_files/save_dir",
                     trials=["small_dataset1.tif", "small_dataset2.tif"])
    main_widget.open_dataset_thread.wait()
    time.sleep(15)
    assert main_widget.data_handler.shape == [400, 150]
    assert main_widget.data_handler.trials_loaded == ["small_dataset1.tif",
                                                      "small_dataset2.tif"]
    assert main_widget.data_handler.trials_all == ["small_dataset1.tif",
                                                   "small_dataset2.tif"]
    assert main_widget.data_handler.trials_loaded_time_trace_indices == [0, 1]
    assert main_widget.data_handler._trials_loaded_indices == [0, 1]
    try:
        app.quit()
    except:
        pass
    shutil.rmtree("test_files/save_dir")
