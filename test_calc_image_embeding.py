from calc_image_embeding import *
import numpy as np
def test_reshape():
    volume = np.array([[[1,1],[1,1]],[[2,2],[2,2]], [[3,3],[3,3]]])
    assert reshape_to_2d_over_time(volume).all() == np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]).all()