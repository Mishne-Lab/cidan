from dask import delayed
class SpatialBox:
    def __init__(self, box_num: int, total_boxes: int, image_shape: int,
                 spatial_overlap: int):
        # TODO implement spatial overlap
        self.box_num = box_num
        self.total_boxes = total_boxes
        self.image_shape = image_shape
        self.boxes_per_row = int(total_boxes ** .5)
        self.y_box_num = box_num // self.boxes_per_row
        self.x_box_num = box_num - (self.y_box_num * self.boxes_per_row)

        self.box_cord_1 = [((image_shape[1] // self.boxes_per_row) *
                           self.x_box_num)-spatial_overlap,
                           (image_shape[2] // self.boxes_per_row) *
                           self.y_box_num-spatial_overlap]
        self.box_cord_2 = [(image_shape[1] // self.boxes_per_row) * (
                self.x_box_num+1)+spatial_overlap,
                           (image_shape[2] // self.boxes_per_row ) * (
                                   self.y_box_num+1)+spatial_overlap]
        self.box_cord_1[0] = 0 if self.box_cord_1[0]<0 else self.box_cord_1[0]
        self.box_cord_1[1] =  0 if self.box_cord_1[1] < 0 \
            else self.box_cord_1[1]
        self.box_cord_2[0] = image_shape[1] if self.box_cord_2[0] > image_shape[1] \
            else \
            self.box_cord_2[0]
        self.box_cord_2[1] = image_shape[2] if self.box_cord_2[1] > image_shape[2] \
            else self.box_cord_2[1]
        self.shape = (image_shape[0],self.box_cord_2[0]-self.box_cord_1[0],
                                  self.box_cord_2[
            1]-self.box_cord_1[1])
        print(box_num,self.shape)
    @delayed
    def extract_box(self, dataset):
        return dataset[:,self.box_cord_1[0]:self.box_cord_2[0], self.box_cord_1[1]:
                       self.box_cord_2[1]]
    @delayed
    def redefine_spatial_cord_2d(self, cord_list):
        return [(x+self.box_cord_1[0],y+self.box_cord_1[1]) for x,y in cord_list]
    @delayed
    def redefine_spatial_cord_1d(self, cord_list):
        box_length = self.box_cord_2[1]-self.box_cord_1[1]
        def change_1_cord(x):
            return ((x//box_length)+self.box_cord_1[0])*self.image_shape[
                2]+self.box_cord_1[1]+x%box_length
        return list(map(change_1_cord,cord_list))

    def convert_1d_to_2d(self):
        # TODO implement this
        pass
if __name__ == '__main__':
    test = SpatialBox(box_num=0,total_boxes=9, image_shape=[1,9,9],spatial_overlap=0)
    pixel_list = test.redefine_spatial_cord_1d([0,4,8]).compute()
    import numpy as np
    zeros = np.zeros((9*9))
    zeros[pixel_list]=1
    print(zeros.reshape((9,9)))
