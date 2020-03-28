
def roi_view_click(main_widget,roi_list, event):
    event.accept()
    pos = event.pos()
    x=int(pos.x())
    y=int(pos.y())
    pixel_with_rois_flat = main_widget.data_handler.pixel_with_rois_flat
    shape = main_widget.data_handler.dataset.shape
    roi_num = int(pixel_with_rois_flat[shape[2] * x + y])
    # TODO change to int
    if roi_num != 0:
        roi_list.set_current_select(roi_num)






