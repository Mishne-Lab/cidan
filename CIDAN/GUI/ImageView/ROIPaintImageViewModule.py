import logging

import numpy as np
from qtpy import QtCore

from CIDAN.GUI.ImageView.ROIImageViewModule import ROIImageViewModule

logger1 = logging.getLogger("CIDAN.ImageView.ROIImageViewModule")


class ROIPaintImageViewModule(ROIImageViewModule):
    def __init__(self, main_widget, tab, settings_tab=True):
        super(ROIPaintImageViewModule, self).__init__(main_widget, tab, settings_tab)
        # part about selecting pixels

        self.select_pixel_on = False  # whether you can currently select pixels in the image
        self.brush_size = 1
        self.current_selected_pixels_list = []  # list of currently selected pixels in their 1d number format
        self.current_selected_pixels_mask = np.zeros((self.data_handler.shape[0],
                                                      self.data_handler.shape[
                                                          1]),
                                                     dtype=bool)  # mask 1 when selected 0 when not
        self.shape = self.data_handler.shape
        self.select_image_flat = np.zeros([self.shape[0] * self.shape[1], 3])
        self.select_pixel_color = [0, 255, 0]
        self.select_mode = "add"  # possibilities add and subtract from current
        # selection

    def roi_view_click(self, event):
        if event.button() == QtCore.Qt.RightButton:
            if self.image_item.raiseContextMenu(event):
                event.accept()
        try:
            if hasattr(self.main_widget.data_handler,
                       "pixel_with_rois_flat") and self.main_widget.data_handler.pixel_with_rois_flat is not None:
                pos = event.pos()

                y = int(pos.x())
                x = int(pos.y())
                if self.select_mode == "magic":
                    self.magic_wand(x, y)
                    print("Done generating ROI")
                elif self.select_pixel_on:
                    event.accept()
                    self.pixel_paint(x, y)

                else:
                    super().roi_view_click(event)
        except ValueError as e:
            if "shape" in e.args[0]:
                self.reset_view()

    def roi_view_drag(self, event):
        # if event.button() == QtCore.Qt.RightButton:
        #     if self.image_item.raiseContextMenu(event):
        #         event.accept()
        if hasattr(self.main_widget.data_handler,
                   "pixel_with_rois_flat") and self.main_widget.data_handler.pixel_with_rois_flat is not None:
            pos = event.pos()

            y = int(pos.x())
            x = int(pos.y())
            if self.select_pixel_on and self.select_mode != "magic":
                event.accept()
                self.pixel_paint(x, y)

    def magic_wand(self, x, y):
        shape = self.data_handler.shape
        # self.clearPixelSelection(update_display=False)
        print("Generating ROI")
        new_roi = self.data_handler.genRoiFromPoint((x, y))
        if len(new_roi) == 0:
            print(
                "Please try again with a bigger growth factor or a different point, we couldn't find an roi where you last selected")
            return False

        for cord_1d in new_roi:
            x_new, y_new = cord_1d // shape[1], cord_1d - (
                    cord_1d // shape[1]) * shape[1]
            self.image_item.image[x_new, y_new] += [0, 255, 0]
            self.current_selected_pixels_list.append(
                shape[1] * x_new + y_new)
            self.current_selected_pixels_mask[x_new, y_new] = True
        self.image_item.updateImage()
        return True
    def pixel_paint(self, x, y):
        try:
            shape = self.main_widget.data_handler.shape
            if self.select_mode == "add":

                for x_dif in range(self.brush_size * 2 + 1):
                    for y_dif in range(self.brush_size * 2 + 1):
                        x_new = x - self.brush_size - 1 + x_dif
                        y_new = y - self.brush_size - 1 + y_dif
                        if shape[1] * x_new + y_new \
                                not in self.current_selected_pixels_list:
                            self.image_item.image[x_new, y_new] += [0, 255, 0]
                            self.current_selected_pixels_list.append(
                                shape[1] * x_new + y_new)
                            self.current_selected_pixels_mask[x_new, y_new] = True

            if self.select_mode == "subtract":
                for x_dif in range(self.brush_size * 2 + 1):
                    for y_dif in range(self.brush_size * 2 + 1):
                        x_new = x - self.brush_size - 1 + x_dif
                        y_new = y - self.brush_size - 1 + y_dif
                        if shape[1] * x_new + y_new \
                                in self.current_selected_pixels_list:
                            self.image_item.image[x_new, y_new] -= [0, 255, 0]
                            self.current_selected_pixels_list.remove(
                                shape[1] * x_new + y_new)
                            self.current_selected_pixels_mask[x_new, y_new] = False

            self.image_item.updateImage()
        except IndexError:
            pass
        except ValueError as e:
            if "shape" in e.args[0]:
                print("Error please try again")
                self.reset_view()

        pass  # TODO use slicing to update pixel based on current thing

    def updateImageDisplay(self, new=False, update=False):
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
                self.clearPixelSelection(update_display=False)
            else:
                self.image_item.image = background_image_scaled_3_channel.reshape(
                    (shape[0], shape[1], 3))
                self.image_item.updateImage(autoLevels=True)

                # if self.add_image:
                combined = (self.roi_image_flat + self.select_image_flat).reshape(
                    (shape[0], shape[1], 3))
                self.image_item.image += combined
                self.image_item.image[
                    self.current_selected_pixels_mask] += self.select_pixel_color

                # else:
                #     combined = self.select_image_flat+self.roi_image_flat
                #     combined_reshaped = combined.reshape((shape[1], shape[2], 3))
                #     mask = np.any(combined != [0, 0, 0], axis=1).reshape((shape[1], shape[2]))
                #
                #     self.image_item.image[mask] = combined_reshaped[mask]

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
                # print("Error please try again")
                self.reset_view()

    def clearPixelSelection(self, update_display=True):
        shape = self.main_widget.data_handler.shape
        self.current_selected_pixels_mask = np.zeros([shape[0], shape[1]], dtype=bool)
        self.current_selected_pixels_list = []
        if update_display:
            self.updateImageDisplay()

    def check_pos_in_image(self, x, y):
        pass
        # TODO add in way to check if in image

    def setBrushSize(self, size):
        """
        Sets the brush size

        self.brush_size is the additional size on all dimensions in addition to middle
        point
        Parameters
        ----------
        size from option input

        Returns
        -------
        nothing
        """
        self.brush_size = int((int(size) - 1) / 2)

    def setSelectorBrushType(self, type):

        if type == "off":
            self.select_pixel_on = False
            self.select_mode = type
        else:
            self.select_pixel_on = True
            self.select_mode = type

    def reset_view(self):
        if not any([x.isRunning() for x in
                    self.main_widget.thread_list]) and not self.resetting_view:
            super().reset_view(updateDisplay=False)
            self.resetting_view = True
            self.select_image_flat = np.zeros(
                [self.data_handler.shape[0] * self.data_handler.shape[1], 3])

            self.clearPixelSelection()
            self.updateImageDisplay(new=True)
            self.resetting_view = False
