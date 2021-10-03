import fire
import numpy as np

from cidan.LSSC.functions.data_manipulation import load_tif_stack, filter_stack


def create_images(input, output):
    for med_on in [True, False]:
        stack = filter_stack(stack=load_tif_stack(path=input), median_filter=med_on,
                             median_filter_size=[3, 3, 3],
                             z_score=False, hist_eq=False, localSpatialDenoising=False
                             )
        max = np.max(stack, axis=0)
        print(np.min(max), np.max(max), np.min(max.astype(np.uint16)),
              np.max(max.astype(np.uint16)))
        np.save(output[:-4] + ("_median" if med_on else "") + "_max.npy", max)
        np.save(output[:-4]+("_median" if med_on else "")+"_mean.npy",np.mean(stack,axis=0))
        np.save(output[:-4]+("_median" if med_on else "")+"_median.npy",np.median(stack,axis=0))

    print("done"+input)
if __name__ == '__main__':
    fire.Fire(create_images)