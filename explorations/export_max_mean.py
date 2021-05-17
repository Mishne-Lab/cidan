import fire
from PIL import Image

from cidan.LSSC.functions.data_manipulation import load_filter_tif_stack
import matplotlib.pyplot as plt
import numpy as np

def create_images(input, output):
    for med_on in [True, False]:
        size, stack =load_filter_tif_stack(path=input, filter=med_on, median_filter=med_on,
                              median_filter_size=[3,3,3],
                              z_score=False, slice_stack=False,
                              slice_every=0, slice_start=0, crop_stack=False,
                              crop_x=[0,0], crop_y=[0,0],
                              load_into_mem = True, trial_split=False,
                              trial_split_length=100, trial_num=0, zarr_path=False)
        max = np.max(stack,axis=0)
        print(np.min(max), np.max(max),np.min(max.astype(np.uint16)), np.max(max.astype(np.uint16)))
        np.save(output[:-4]+("_median" if med_on else "")+"_max.npy",max)
        np.save(output[:-4]+("_median" if med_on else "")+"_mean.npy",np.mean(stack,axis=0))
        np.save(output[:-4]+("_median" if med_on else "")+"_median.npy",np.median(stack,axis=0))

    print("done"+input)
if __name__ == '__main__':
    fire.Fire(create_images)