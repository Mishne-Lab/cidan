import logging
from typing import Tuple

import numpy as np
from dask import delayed

logger1 = logging.getLogger("CIDAN.LSSC.SpatialBox")


class SpatialBox:
    def __init__(self, box_num: int, total_boxes: int, image_shape: Tuple[int, int],
                 spatial_overlap: int):
        logger1.debug(
            "Spatial Box creation inputs: box num {0}, total boxes {1}, image shape {2}, spatial overlap {3}".format(
                box_num, total_boxes, image_shape, spatial_overlap))
        # TODO implement spatial overlap
        self.box_num = box_num
        self.total_boxes = total_boxes
        self.image_shape = image_shape
        self.boxes_per_row = int(total_boxes ** .5)
        self.spatial_overlap = spatial_overlap
        self.y_box_num = box_num // self.boxes_per_row
        self.x_box_num = box_num - (self.y_box_num * self.boxes_per_row)

        self.box_cord_1 = [((image_shape[0] // self.boxes_per_row) *
                            self.x_box_num) - spatial_overlap,
                           (image_shape[1] // self.boxes_per_row) *
                           self.y_box_num - spatial_overlap]
        self.box_cord_2 = [(image_shape[0] // self.boxes_per_row) * (
                self.x_box_num + 1) + spatial_overlap,
                           (image_shape[1] // self.boxes_per_row) * (
                                   self.y_box_num + 1) + spatial_overlap]

        self.box_cord_1[0] = 0 if self.box_cord_1[0] < 0 else self.box_cord_1[0]
        self.box_cord_1[1] = 0 if self.box_cord_1[1] < 0 \
            else self.box_cord_1[1]
        self.box_cord_2[0] = image_shape[0] if self.box_cord_2[0] > image_shape[0] \
            else \
            self.box_cord_2[0]
        self.box_cord_2[1] = image_shape[1] if self.box_cord_2[1] > image_shape[1] \
            else self.box_cord_2[1]
        self.shape = (self.box_cord_2[0] - self.box_cord_1[0],
                      self.box_cord_2[
                          1] - self.box_cord_1[1])
        logger1.debug(("Spatial box creation: Boxes per row {0}, y_box_num {1}, " +
                       "x_box_num" + " {2}, box cord 1 {3}, box cord 2 {4}, shape {5}"
                       ).format(self.boxes_per_row, self.y_box_num, self.x_box_num,
                                self.box_cord_1, self.box_cord_2, self.shape))

    @delayed
    def extract_box(self, dataset):
        return dataset[:, self.box_cord_1[0]:self.box_cord_2[0], self.box_cord_1[1]:
                                                                 self.box_cord_2[1]]

    @delayed
    def redefine_spatial_cord_2d(self, cord_list):
        return [(x + self.box_cord_1[0], y + self.box_cord_1[1]) for x, y in cord_list]

    def pointInBox(self, point):
        """
        Checks if a point is in the box
        Parameters
        ----------
        point

        Returns
        -------

        """
        return self.box_cord_1[0] <= point[0] <= self.box_cord_2[0] \
               and self.box_cord_1[1] <= point[1] <= self.box_cord_2[1]

    def point_to_box_point(self, point):
        """
        Converts a point in the image to its cords in the box
        Parameters
        ----------
        point 2d point

        Returns
        -------
        tuple of new cords
        """
        return (point[0] - self.box_cord_1[0], point[1] - self.box_cord_1[1])

    @delayed
    def redefine_spatial_cord_1d(self, cord_list):
        box_length = self.box_cord_2[1] - self.box_cord_1[1]

        def change_1_cord(x):
            return ((x // box_length) + self.box_cord_1[0]) * self.image_shape[
                1] + self.box_cord_1[1] + x % box_length

        return list(map(change_1_cord, cord_list))

    def convert_1d_to_2d(self, cord_list):
        def change_1_cord(cord_1d):
            return int(cord_1d // self.shape[1]), int(cord_1d - (
                    cord_1d // self.shape[1]) * self.shape[1])

        return list(map(change_1_cord, cord_list))

    def data_w_out_spatial_overlap(self, data):
        """

        Parameters
        ----------
        data 2d dataset

        Returns
        -------

        """
        if self.total_boxes == 1:
            return data
        x = [0, self.shape[0]]
        y = [0, self.shape[1]]
        # This uses which column each box is in to determin overlap parts, first, last,
        # any other column
        if self.box_num % self.boxes_per_row == 0:
            x[1] = self.shape[0] - self.spatial_overlap
        elif self.box_num % self.boxes_per_row == self.boxes_per_row - 1:
            x[0] = self.spatial_overlap
        else:
            x[0] = self.spatial_overlap
            x[1] = self.shape[0] - self.spatial_overlap
        # This uses which row each box is in to determin overlap parts, first, last,
        # any other row
        if self.box_num // self.boxes_per_row == 0:
            y[1] = self.shape[1] - self.spatial_overlap
        elif self.box_num // self.boxes_per_row == self.boxes_per_row - 1:
            y[0] = self.spatial_overlap
        else:
            y[0] = self.spatial_overlap
            y[1] = self.shape[1] - self.spatial_overlap
        return data[x[0]:x[1], y[0]:y[1]]


def combine_images(spatial_box_list, data_list):
    """

    Parameters
    ----------
    spatial_box_list
    data_list list of reshaped eigen vectors in the correct shape for these boxes

    Returns
    -------

    """
    # Going to go through the lists in a 2d format using number of spatial boxes is always square
    spatial_box_num = len(spatial_box_list)
    spatial_box_root = int(spatial_box_num ** .5)
    data_matched = []
    for y in range(spatial_box_root):
        temp = []
        for x in range(spatial_box_root):
            current_spatial_box = spatial_box_list[y * spatial_box_root + x]
            current_data = data_list[y * spatial_box_root + x]
            temp.append(current_spatial_box.data_w_out_spatial_overlap(current_data))
        stacked = np.vstack(temp)
        data_matched.append(stacked)
    all_data = np.hstack(data_matched)

    return all_data


if __name__ == '__main__':
    test = SpatialBox(box_num=0, total_boxes=9, image_shape=[1, 9, 9],
                      spatial_overlap=0)
    pixel_list = test.redefine_spatial_cord_1d([0, 4, 8]).compute()

    zeros = np.zeros((9 * 9))
    zeros[pixel_list] = 1
    print(zeros.reshape((9, 9)))
