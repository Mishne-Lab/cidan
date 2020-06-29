import logging

import numpy as np
from qtpy import QtCore
from qtpy.QtCore import *
from qtpy.QtWidgets import *

from CIDAN.GUI.ImageView.ImageViewModule import ImageViewModule
from CIDAN.GUI.Inputs.OptionInput import OptionInput

logger1 = logging.getLogger("CIDAN.ImageView.ROIImageViewModule")


class ROIImageViewModule(ImageViewModule):
    # QApplication.mouseButtons() == Qt.LeftButton
    def __init__(self, main_widget, tab, settings_tab=True):
        super(ROIImageViewModule, self).__init__(main_widget, histogram=False)
        self.tab = tab
        self.resetting_view = False  # Way to prevent infinite loops of reset_view
        self.current_foreground_intensity = 80
        self.click_event = False
        self.outlines = True
        self.trial_selector_input = OptionInput(
            "Select Trial for background image(Min/Max only)", "", self.set_background,
            val_list=self.data_handler.trials_loaded,
            tool_tip="Select Trial to display",
            display_tool_tip=False, default_index=0,
            show_name=True)
        self.set_background("", "Max Image", update_image=False)
        self.image_item.mouseClickEvent = lambda x: self.roi_view_click(x)
        self.image_item.mouseDragEvent = lambda x: self.roi_view_drag(x)

        shape = main_widget.data_handler.shape
        self.select_image_flat = np.zeros([shape[0] * shape[1], 3])
        self.box_selector_enabled = False
        self.box_selector_cords = [(0, 0), (0, 0)]
        self.current_background_name = "Max Image"

        if settings_tab:
            self.layout.removeWidget(self.image_view)
            self.layout.addWidget(self.createTabLayout())

    def createTabLayout(self):
        # ROI view tab section
        roi_view_tabs = QTabWidget()
        roi_view_tabs.setStyleSheet("QTabWidget {font-size: 20px;}")
        # Display settings tab
        self.display_settings_layout = QVBoxLayout()

        display_settings = QWidget()
        display_settings.setLayout(self.display_settings_layout)
        image_chooser = OptionInput("ROI Display type::", "",
                                    on_change_function=self.set_image,
                                    default_index=0,
                                    tool_tip="Choose background to display",
                                    val_list=["Outlines", "Blob"])

        self.display_settings_layout.addWidget(image_chooser)

        self.background_chooser = OptionInput("Background:", "",
                                              on_change_function=self.set_background,
                                              default_index=2,
                                              tool_tip="Choose background to display",
                                              val_list=["Blank Image", "Mean Image",
                                                        "Max Image",
                                                        # "Temporal Correlation Image",
                                                        "Eigen Norm Image"])

        self.display_settings_layout.addWidget(self.background_chooser)
        self.display_settings_layout.addWidget(self.trial_selector_input)

        background_slider_layout = QHBoxLayout()
        background_slider_layout.addWidget(QLabel("0"))
        # initializes a slider to control how much to blend background image in when
        # blob is view is enabled
        self.background_slider = QSlider(Qt.Horizontal)
        self.background_slider.setMinimum(0)
        self.background_slider.setMaximum(100)
        self.background_slider.setSingleStep(1)
        self.background_slider.valueChanged.connect(
            self.intensitySliderChanged)
        try:
            self.background_slider.setValue(
                self.current_foreground_intensity)
        except AttributeError:
            pass
        background_slider_layout.addWidget(self.background_slider)
        background_slider_layout.addWidget(QLabel("10"))
        self.display_settings_layout.addWidget(QLabel("Change overlay intensity:"))
        self.display_settings_layout.addLayout(background_slider_layout)

        # ROI image view part
        self.setStyleSheet(
            "margin:0px; border:0px  solid rgb(50, 65, "
            "75); padding: 0px;")
        roi_view_widget = QWidget()
        roi_view_widget_layout = QVBoxLayout()
        roi_view_widget_layout.setContentsMargins(0, 0, 0, 0)
        roi_view_widget_layout.addWidget(self.image_view)
        roi_view_widget.setLayout(roi_view_widget_layout)
        roi_view_tabs.addTab(roi_view_widget, "ROI Display")
        roi_view_tabs.addTab(display_settings, "Display Settings")
        return roi_view_tabs

    def settingsLayout(self):
        return self.display_settings_layout

    def intensitySliderChanged(self):
        self.current_foreground_intensity = 10 - (
                float(self.background_slider.value()) / 10)

        self.updateImageDisplay()

    def set_background(self, name, func_name, update_image=True):
        if (self.main_widget.checkThreadRunning()):

            # Background refers to the image behind the rois
            shape = self.main_widget.data_handler.shape
            if func_name == "Mean Image":
                self.current_background = \
                    self.main_widget.data_handler.mean_images[
                        self.data_handler.trials_loaded.index(
                            self.trial_selector_input.current_state())][:].reshape(
                    [-1, 1])
            elif func_name == "Max Image":
                self.current_background = self.main_widget.data_handler.max_images[
                                              self.data_handler.trials_loaded.index(
                                                  self.trial_selector_input.current_state())][
                                          :].reshape(
                    [-1, 1])
            elif func_name == "Blank Image":
                self.current_background = np.zeros([shape[0] * shape[1], 1])
            # if func_name == "Temporal Correlation Image":
            #     self.current_background = self.data_handler.temporal_correlation_image.reshape(
            #         [-1, 1])
            elif func_name == "Eigen Norm Image":
                self.current_background = self.data_handler.eigen_norm_image.reshape(
                    [-1, 1])
            else:
                self.current_background_name = "Max Image"
                self.current_background = self.main_widget.data_handler.max_images[
                                              self.data_handler.trials_loaded.index(
                                                  self.trial_selector_input.current_state())][
                                          :].reshape(
                    [-1, 1])
            if update_image:
                self.updateImageDisplay()

    def set_image(self, name, func_name, update_image=True):
        if (self.main_widget.checkThreadRunning() and self.data_handler.rois_loaded):

            # Background refers to the image behind the rois
            shape = self.main_widget.data_handler.edge_roi_image_flat.shape
            if func_name == "Outlines":
                self.outlines = True
                self.roi_image_flat = np.hstack([self.data_handler.edge_roi_image_flat,
                                                 np.zeros(shape),
                                                 np.zeros(shape)])
            if func_name == "Blob":
                self.outlines = False
                self.roi_image_flat = self.main_widget.data_handler.pixel_with_rois_color_flat

            if update_image:
                self.updateImageDisplay()

    def updateImageDisplay(self, new=False):

        try:
            # new is to determine whether the zoom should be saved
            # TODO add in update with image paint layer
            shape = self.main_widget.data_handler.shape
            # range_list = self.main_widget.roi_image_view.image_view.view.viewRange()
            background_max = self.current_background.max()
            background_image_scaled = (self.current_foreground_intensity * 255 / (
                background_max if background_max != 0 else 1)) * self.current_background
            background_image_scaled_3_channel = np.hstack(
                [background_image_scaled, background_image_scaled,
                 background_image_scaled])
            if new and not hasattr(self.main_widget.data_handler,
                                   "edge_roi_image_flat"):
                self.image_item.image = background_image_scaled_3_channel.reshape(
                    (shape[0], shape[1], 3))
                self.image_item.updateImage(autoLevels=True)
            elif new:
                # if self.add_image:
                combined = self.roi_image_flat + background_image_scaled_3_channel + self.select_image_flat

                # else:
                #     combined = background_image_scaled + self.select_image_flat
                #     mask = np.any(self.roi_image_flat != [0, 0, 0], axis=1)
                #     combined[mask] = self.roi_image_flat[mask]
                combined_reshaped = combined.reshape((shape[0], shape[1], 3))
                self.tab.image_view.setImage(combined_reshaped)
            else:
                self.image_item.image = background_image_scaled_3_channel.reshape(
                    (shape[0], shape[1], 3))
                self.image_item.updateImage(autoLevels=True)

                # if self.add_image:
                combined = (self.roi_image_flat + self.select_image_flat).reshape(
                    (shape[0], shape[1], 3))
                self.image_item.image += combined

                self.image_item.updateImage(autoLevels=False)

                # self.main_widget.roi_image_view.image_view.view.setRange(xRange=range_list[0],
                #                                                      yRange=range_list[1])
                # range_list = self.main_widget.roi_image_view.image_view.view.viewRange()
                # print(range_list)

            pass
        except AttributeError as e:
            logger1.error(e)

        except ValueError as e:
            if "shape" in e.args[0]:
                self.reset_view()

    def selectRoi(self, num):
        try:
            color_select = (245, 249, 22)
            color_roi = self.main_widget.data_handler.color_list[
                (num - 1) % len(self.main_widget.data_handler.color_list)]
            self.select_image_flat[
                self.main_widget.data_handler.rois[num - 1]] = color_select
            self.updateImageDisplay()
        except AttributeError:
            pass
        except ValueError as e:
            if "shape" in e.args[0]:
                print("Error please try again")
                self.reset_view()

    def deselectRoi(self, num):
        try:
            color = self.main_widget.data_handler.color_list[
                (num - 1) % len(self.main_widget.data_handler.color_list)]
            shape_flat = self.data_handler.edge_roi_image_flat.shape
            self.select_image_flat[self.main_widget.data_handler.rois[
                num - 1]] = color if not self.outlines \
                else np.hstack([self.data_handler.edge_roi_image_flat,
                                np.zeros(shape_flat),
                                np.zeros(shape_flat)])[
                self.main_widget.data_handler.rois[num - 1]]
            self.updateImageDisplay()
        except ValueError as e:
            if "shape" in e.args[0]:
                print("Error please try again")
                self.reset_view()

    def zoomRoi(self, num):
        """
        Zooms in to a certain roi
        Parameters
        ----------
        num : int
            roi num starts at 1

        Returns
        -------
        Nothing
        """
        num = num - 1

        max_cord = self.main_widget.data_handler.roi_max_cord_list[num] + 15

        min_cord = self.main_widget.data_handler.roi_min_cord_list[num] - 15

        self.image_view.getView().setXRange(min_cord[1],
                                            max_cord[1])
        self.image_view.getView().setYRange(min_cord[0],
                                            max_cord[0])

    def roi_view_click(self, event):
        if event.button() == QtCore.Qt.RightButton:
            if self.image_item.raiseContextMenu(event):
                event.accept()
        if hasattr(self.main_widget.data_handler,
                   "pixel_with_rois_flat") and self.main_widget.data_handler.pixel_with_rois_flat is not None:
            pos = event.pos()

            y = int(pos.x())
            x = int(pos.y())
            self.click_event = True
            pixel_with_rois_flat = self.main_widget.data_handler.pixel_with_rois_flat
            shape = self.main_widget.data_handler.shape
            roi_num = int(pixel_with_rois_flat[shape[1] * x + y])
            # TODO change to int
            if roi_num != 0:
                event.accept()
                self.tab.roi_list_module.set_current_select(roi_num)

    def roi_view_drag(self, event):
        prev = True
        pos = event.pos()

        y = int(pos.x())  # Because of column order thing in image_view
        x = int(pos.y())
        modifiers = QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier and not self.box_selector_enabled:
            self.box_selector_enabled = True
            self.box_selector_cords = [(x, y), (x, y)]
            prev = False
        if self.box_selector_enabled:
            event.accept()
            if prev:
                try:

                    if (self.box_selector_cords[0][0] < self.box_selector_cords[1][0]):
                        self.image_item.image[
                        self.box_selector_cords[0][0]:self.box_selector_cords[1][0],
                        self.box_selector_cords[0][1]] -= [255, 255, 255]
                        self.image_item.image[
                        self.box_selector_cords[0][0]:self.box_selector_cords[1][0],
                        self.box_selector_cords[1][1]] -= [255, 255, 255]
                    else:
                        self.image_item.image[
                        self.box_selector_cords[1][0]:self.box_selector_cords[0][0],
                        self.box_selector_cords[0][1]] -= [255, 255, 255]
                        self.image_item.image[
                        self.box_selector_cords[1][0]:self.box_selector_cords[0][0],
                        self.box_selector_cords[1][1]] -= [255, 255, 255]
                    if self.box_selector_cords[0][1] < self.box_selector_cords[1][1]:
                        self.image_item.image[self.box_selector_cords[0][0],
                        self.box_selector_cords[0][1]:self.box_selector_cords[1][1]] -= \
                            [255, 255, 255]
                        self.image_item.image[self.box_selector_cords[1][0],
                        self.box_selector_cords[0][1]:self.box_selector_cords[1][1]] -= \
                            [255, 255, 255]
                    else:
                        self.image_item.image[self.box_selector_cords[0][0],
                        self.box_selector_cords[1][1]:self.box_selector_cords[0][1]] -= \
                            [255, 255, 255]
                        self.image_item.image[self.box_selector_cords[1][0],
                        self.box_selector_cords[1][1]:self.box_selector_cords[0][1]] -= \
                            [255, 255, 255]
                except IndexError:
                    pass
            if QApplication.mouseButtons() != Qt.LeftButton:
                self.box_selector_enabled = False
                shape = self.main_widget.data_handler.shape
                rois_image = np.reshape(
                    self.main_widget.data_handler.pixel_with_rois_flat, shape)
                if (self.box_selector_cords[0][0] < self.box_selector_cords[1][0]):
                    rois_image = rois_image[self.box_selector_cords[0][0]:
                                            self.box_selector_cords[1][0]]
                else:
                    rois_image = rois_image[self.box_selector_cords[1][0]:
                                            self.box_selector_cords[0][0]]
                if self.box_selector_cords[0][1] < self.box_selector_cords[1][1]:
                    rois_image = rois_image[:, self.box_selector_cords[0][1]:
                                               self.box_selector_cords[1][1]]
                else:
                    rois_image = rois_image[:, self.box_selector_cords[1][1]:
                                               self.box_selector_cords[0][1]]
                rois_selected = np.unique(rois_image)[1:]
                self.tab.update_time = False
                for x in rois_selected:
                    self.tab.roi_list_module.roi_item_list[
                        int(x) - 1].check_box.setChecked(True)
                self.tab.update_time = True
                self.tab.deselectRoiTime()
                self.box_selector_cords = [(0, 0), (0, 0)]

            else:
                self.box_selector_cords[1] = (x, y)
                try:
                    if (self.box_selector_cords[0][0] < self.box_selector_cords[1][0]):
                        self.image_item.image[
                        self.box_selector_cords[0][0]:self.box_selector_cords[1][0],
                        self.box_selector_cords[0][1]] += [255, 255, 255]
                        self.image_item.image[
                        self.box_selector_cords[0][0]:self.box_selector_cords[1][0],
                        self.box_selector_cords[1][1]] += [255, 255, 255]
                    else:
                        self.image_item.image[
                        self.box_selector_cords[1][0]:self.box_selector_cords[0][0],
                        self.box_selector_cords[0][1]] += [255, 255, 255]
                        self.image_item.image[
                        self.box_selector_cords[1][0]:self.box_selector_cords[0][0],
                        self.box_selector_cords[1][1]] += [255, 255, 255]
                    if self.box_selector_cords[0][1] < self.box_selector_cords[1][1]:
                        self.image_item.image[self.box_selector_cords[0][0],
                        self.box_selector_cords[0][1]:self.box_selector_cords[1][1]] += \
                            [255, 255, 255]
                        self.image_item.image[self.box_selector_cords[1][0],
                        self.box_selector_cords[0][1]:self.box_selector_cords[1][1]] += \
                            [255, 255, 255]
                    else:
                        self.image_item.image[self.box_selector_cords[0][0],
                        self.box_selector_cords[1][1]:self.box_selector_cords[0][1]] += \
                            [255, 255, 255]
                        self.image_item.image[self.box_selector_cords[1][0],
                        self.box_selector_cords[1][1]:self.box_selector_cords[0][1]] += \
                            [255, 255, 255]

                except IndexError:
                    pass
            self.image_item.updateImage()

    def reset_view(self, updateDisplay=True):
        if not any([x.isRunning() for x in
                    self.main_widget.thread_list]) and not self.resetting_view:
            self.resetting_view = True
            if hasattr(self.main_widget.data_handler,
                       "edge_roi_image_flat") and self.main_widget.data_handler.edge_roi_image_flat is not None:
                shape = self.main_widget.data_handler.edge_roi_image_flat.shape
                self.data_handler.save_rois(self.data_handler.rois)
                self.select_image_flat = np.zeros([shape[0] * shape[1], 3])

                if self.outlines:
                    self.roi_image_flat = np.hstack(
                        [self.data_handler.edge_roi_image_flat,
                         np.zeros(shape),
                         np.zeros(shape)])



                else:
                    self.roi_image_flat = self.main_widget.data_handler.pixel_with_rois_color_flat
            self.set_background("", self.current_background_name, update_image=False)
            if (updateDisplay):
                self.updateImageDisplay(new=True)
            self.resetting_view = False
