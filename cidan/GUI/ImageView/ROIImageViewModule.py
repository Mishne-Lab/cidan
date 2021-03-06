import logging

import numpy as np
from qtpy import QtCore
from qtpy.QtCore import *
from qtpy.QtWidgets import *

from cidan.GUI.ImageView.ImageViewModule import ImageViewModule
from cidan.GUI.Inputs.OptionInput import OptionInput

logger1 = logging.getLogger("cidan.ImageView.ROIImageViewModule")


class ROIImageViewModule(ImageViewModule):
    # QApplication.mouseButtons() == Qt.LeftButton
    def __init__(self, main_widget, tab, settings_tab=True):
        super(ROIImageViewModule, self).__init__(main_widget, histogram=False)
        self.tab = tab
        self.resetting_view = False  # Way to prevent infinite loops of reset_view
        self.current_foreground_intensity = 30
        self.click_event = False
        self.outlines = True
        self.trial_selector_input = OptionInput(
            "Time Block:", "",
            lambda x, y: self.set_background("", self.current_background_name),
            val_list=self.data_handler.trials_loaded,
            tool_tip="Select Time block to display",
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

    def createSettings(self):
        self.display_settings_layout = QVBoxLayout()

        display_settings = QWidget()
        display_settings.setLayout(self.display_settings_layout)
        image_chooser = OptionInput("ROI Display type:", "",
                                    on_change_function=self.set_image,
                                    default_index=0,
                                    tool_tip="Choose background to display",
                                    val_list=["Outlines", "Blob", "Neuropil"])

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
        label_0 = QLabel("0")
        label_0.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        background_slider_layout.addWidget(label_0)
        # initializes a slider to control how much to blend background image in when
        # blob is view is enabled
        self.background_slider = QSlider(Qt.Horizontal)
        self.background_slider.setStyleSheet("QWidget {border: 0px solid #32414B;}")
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
        label_10 = QLabel("10")
        label_10.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        background_slider_layout.addWidget(label_10)
        label_overlay = QLabel("Change background intensity:")
        label_overlay.setStyleSheet("QWidget {border: 0px solid #32414B;}")
        self.display_settings_layout.addWidget(label_overlay)
        self.display_settings_layout.addLayout(background_slider_layout)
        return display_settings
    def createTabLayout(self):
        # ROI view tab section
        roi_view_tabs = QTabWidget()
        roi_view_tabs.setStyleSheet("QTabWidget {font-size: 20px;}")
        # Display settings tab
        display_settings = self.createSettings()

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
        self.current_foreground_intensity = (
                float(self.background_slider.value()) / 10)

        self.updateImageDisplay()

    def set_background(self, name, func_name, update_image=True, reset_override=False):
        if (not self.resetting_view or reset_override):
            self.main_widget.console.updateText(
                "Setting current background to: %s" % func_name)
            if type(self.main_widget.data_handler.mean_images[
                        self.data_handler.trials_loaded.index(
                            self.trial_selector_input.current_state())]) == bool:
                self.data_handler.dataset_trials_filtered_loaded[
                    self.data_handler.trials_loaded.index(
                        self.trial_selector_input.current_state())].compute()
            # Background refers to the image behind the rois
            shape = self.main_widget.data_handler.shape
            if func_name == "Mean Image":
                self.current_background = \
                    self.main_widget.data_handler.mean_images[
                        self.data_handler.trials_loaded.index(
                            self.trial_selector_input.current_state())][:].reshape(
                    [-1, 1])
                self.current_background_name = "Mean Image"
            elif func_name == "Max Image":
                self.current_background = self.main_widget.data_handler.max_images[
                                              self.data_handler.trials_loaded.index(
                                                  self.trial_selector_input.current_state())][
                                          :].reshape(
                    [-1, 1])
                self.current_background_name = "Max Image"
            elif func_name == "Blank Image":
                self.current_background = np.zeros([shape[0] * shape[1], 1])
                self.current_background_name = "Blank Image"
            # if func_name == "Temporal Correlation Image":
            #     self.current_background = self.data_handler.temporal_correlation_image.reshape(
            #         [-1, 1])
            elif func_name == "Eigen Norm Image":
                try:
                    self.current_background = self.data_handler.eigen_norm_image.reshape(
                        [-1, 1])
                    self.current_background_name = "Eigen Norm Image"
                except AttributeError:
                    print("Eigen vectors aren't currently generated or valid")


                    self.current_background = self.main_widget.data_handler.max_images[
                                                  self.data_handler.trials_loaded.index(
                                                      self.trial_selector_input.current_state())][
                                              :].reshape(
                        [-1, 1])
                    if update_image:
                        self.updateImageDisplay()
                    self.main_widget.console.updateText(
                        "Can't display Eigen Norm image, Eigen vectors aren't currently generated or valid")

                    self.current_background_name = "Max Image"
                    return
            else:
                self.current_background_name = "Max Image"
                try:
                    self.trial_selector_input.set_default_val()

                    self.current_background = self.main_widget.data_handler.max_images[
                                              self.data_handler.trials_loaded.index(
                                                  self.trial_selector_input.current_state())][
                                          :].reshape(
                        [-1, 1])
                except AttributeError:
                    pass
            try:
                if self.current_background_name != self.background_chooser.current_state():
                    # pass
                    self.background_chooser.set_val(self.current_background_name)
            except AttributeError:
                pass
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
            if func_name == "Neuropil":
                self.outlines = False
                self.roi_image_flat = self.main_widget.data_handler.neuropil_image_display
            if update_image:
                self.updateImageDisplay()

    def updateImageDisplay(self, new=False):

        try:
            # new is to determine whether the zoom should be saved
            # TODO add in update with image paint layer
            shape = self.main_widget.data_handler.shape
            # range_list = self.main_widget.roi_image_view.image_view.view.viewRange()
            background_max = np.percentile(self.current_background, 98)
            background_min = np.percentile(self.current_background, 2)
            background_image_scaled = (self.current_foreground_intensity / 7 * (
                    self.current_background - background_min) * 255 / (
                                           (background_max - background_min) if (
                                                                                        background_max - background_min) != 0 else 1))
            background_image_scaled_3_channel = np.hstack(
                [background_image_scaled, background_image_scaled,
                 background_image_scaled])
            if new and not hasattr(self.main_widget.data_handler,
                                   "edge_roi_image_flat"):
                self.image_item.image = background_image_scaled_3_channel.reshape(
                    (shape[0], shape[1], 3))
                self.image_item.updateImage(autoLevels=False)
                self.image_item.setLevels((0, 255))
            elif new:
                # if self.add_image:
                combined = self.roi_image_flat * .45 + background_image_scaled_3_channel * .45 + self.select_image_flat * .45

                # else:
                #     combined = background_image_scaled + self.select_image_flat
                #     mask = np.any(self.roi_image_flat != [0, 0, 0], axis=1)
                #     combined[mask] = self.roi_image_flat[mask]
                combined_reshaped = combined.reshape((shape[0], shape[1], 3))
                self.image_item.setLevels((0, 255))

                self.tab.image_view.setImage(combined_reshaped)
            else:
                self.image_item.image = background_image_scaled_3_channel.reshape(
                    (shape[0], shape[1], 3)) * .45
                self.image_item.updateImage(autoLevels=False)
                self.image_item.setLevels((0, 255))


                # if self.add_image:
                combined = (
                            self.roi_image_flat * .45 + self.select_image_flat * .45).reshape(
                    (shape[0], shape[1], 3))
                self.image_item.image += combined

                self.image_item.updateImage(autoLevels=False)
                # pen =mkPen('y', width=3, style=QtCore.Qt.DashLine)
                # if self.outlines:
                #     for roi_contour in self.data_handler.all_roi_contours:
                #         self.plot_item.plot(roi_contour, clear=True)
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
            color_select = (245, 249, 22) if self.outlines else (255, 255, 255)
            color_roi = self.main_widget.data_handler.color_list[
                (num - 1) % len(self.main_widget.data_handler.color_list)]
            self.select_image_flat[
                self.main_widget.data_handler.rois[num - 1]] = color_select
            self.updateImageDisplay()
        except AttributeError:
            pass
        except IndexError:
            self.main_widget.console.updateText(
                "Please regenerate ROIs before trying this operation")

            print("Please regenerate ROIs before trying this operation")
        except ValueError as e:
            if "shape" in e.args[0]:
                self.main_widget.console.updateText("Error please try again")
                print("Error please try again")
                self.reset_view()

    def deselectRoi(self, num):
        try:
            color = self.main_widget.data_handler.color_list[
                (num - 1) % len(self.main_widget.data_handler.color_list)]
            shape_flat = self.data_handler.edge_roi_image_flat.shape
            self.select_image_flat[self.main_widget.data_handler.rois[
                num - 1]] = (0, 0, 0)
            self.updateImageDisplay()
        except ValueError as e:
            if "shape" in e.args[0]:
                self.main_widget.console.updateText("Error please try again")
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

        self.image_view.getView().setYRange(min_cord[1],
                                            max_cord[1])
        self.image_view.getView().setXRange(min_cord[0],
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
            if len(self.trial_selector_input.val_list) != len(
                    self.data_handler.trials_loaded) or any(
                    [x != y for x, y in zip(self.trial_selector_input.val_list,
                                            self.data_handler.trials_loaded)]):
                self.trial_selector_input.set_new_options(
                    self.data_handler.trials_loaded)
                self.trial_selector_input.set_default_val()

            self.set_background("", self.current_background_name, update_image=False,
                                reset_override=True)

            if (updateDisplay):
                self.updateImageDisplay(new=True)
            self.resetting_view = False

    def setImage(self, data):

        # if self.already_loaded == False:
        #     print("changed image")
        #     self.already_loaded = True
        #     self.layout.removeWidget(self.no_image_message)
        #     self.no_image_message.deleteLater()
        #     # self.layout.setAlignment(Qt.AlignLeft)
        #     self.image_view = ImageView()
        #
        #     self.layout.addWidget(self.image_view)
        # bottom_5 = np.percentile(data, 5)
        # top_5 = np.percentile(data, 95)
        # top_10 = np.percentile(data, 90)
        # bottom_2 = np.percentile(data, 2)
        # top_2 = np.percentile(data, 98)
        # data[data>top_10] = top_10
        self.image_view.setImage(data, levelMode='mono', autoRange=True,
                                 autoLevels=True, autoHistogramRange=True)
        # self.top_2 = np.percentile(data, 98)
        # self.bottom_5 = np.percentile(data, 5)
        # self.image_view.setLevels(bottom_5, top_2)
        # self.image_view.setHistogramRange(bottom_2, top_2)
