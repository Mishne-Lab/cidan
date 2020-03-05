from LSSC.Stack import Stack
from LSSC.functions.data_manipulation import load_tif_stack


class Stack_Wrapper:
    """
    This Stack_Wrapper class wraps stack to make it more easily multithreaded
    and split computation around multiple threads
    """

    def __init__(self, file_path, trial_index, output_directory, parameters,
                 num_squares=1, overlap_pixels=(0, 0), num_time_frames=1,
                 overlap_time_steps=0, save_images=False,
                 gen_new=False):
        assert (float(num_squares)**.5).is_integer()
        self.parameters = parameters
        vol = load_tif_stack(file_path)
        shape = vol.shape
        if self.parameters.slice:
            shape[0] = (shape[0] - self.parameters.slice_start) \
                       // self.parameters.slice_every
        num_x_frames = float(num_squares)**.5
        num_y_frames = num_x_frames
        time_frames = self.gen_frames(shape[0], num_time_frames, overlap_time_steps)
        x_frames = self.gen_frames(shape[1], num_x_frames, overlap_pixels)
        y_frames = self.gen_frames(shape[2], num_y_frames, overlap_pixels)
        bounding_boxes = [tuple(zip(*(tuple(x+y+z)))) for x,y,z in
                          (time_frames, x_frames, y_frames)]
        stack_list =

    def gen_frames(self, total_steps, num_frames, overlap):
        return [(
            total_steps// num_frames * x - overlap if
            total_steps // num_frames * x - overlap > 0 else
            total_steps // num_frames * x,
            total_steps // num_frames * x + overlap if
            total_steps // num_frames * x + overlap <
            total_steps else total_steps // num_frames * x) for x in
            range(num_frames)]
    def gen_clusters(self):
        pass

    def gen_embedding_image(self):
        pass
