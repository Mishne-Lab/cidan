import numpy as np

from cidan.LSSC.functions.data_manipulation import load_filter_tif_stack


def test_load_filter_tif_stack():
    test, stack = load_filter_tif_stack(path="test_files/small_dataset1.tif",
                                        filter=True,
                                        median_filter=True, median_filter_size=[3, 3, 3],
                                        z_score=False, slice_stack=False, slice_start=0,
                                        slice_every=3, crop_stack=False, crop_x=[0, 400],
                                        crop_y=[0, 155])
    assert stack.shape == (99, 400, 150)
    assert stack.dtype == np.float32
    test, stack = load_filter_tif_stack(path="test_files/small_dataset1.tif",
                                        filter=True,
                                        median_filter=False, median_filter_size=[3, 3, 3],
                                        z_score=True, slice_stack=True, slice_start=0,
                                        slice_every=3, crop_stack=True, crop_x=[4, 200],
                                        crop_y=[4, 125])
    assert stack.shape == (33, 196, 121)
    assert stack.dtype == np.float32
    mean = np.mean(stack, axis=0)
    assert np.max(mean) < .01
    assert np.min(mean) > -.01
