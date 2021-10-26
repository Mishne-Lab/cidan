import json
import os
import numpy as np
from scipy.io import savemat

import cidan
import fire

from cidan.LSSC.functions.data_manipulation import load_tif_stack, reshape_to_2d_over_time, cord_2d_to_pixel_num
from cidan.TimeTrace.deltaFOverF import calculateDeltaFOverF


def main(roi_json, folder, output):
    files = sorted(os.listdir(folder))
    print("File order: ", files)
    out = {}
    with open(roi_json, "r") as f:
        # test2=f.read()
        test = json.loads(f.read())

    rois = [[(y[0], y[1]) for y in x["coordinates"]] for x in test]

    out = []
    for file in files:
        stack = load_tif_stack(os.path.join(folder,file))
        rois_2d = [cord_2d_to_pixel_num(np.array(roi).transpose(), stack.shape) for roi in
                   rois]
        stack_2d = reshape_to_2d_over_time(stack)
        out_curr = []
        for roi in rois_2d:
            out_curr.append(calculateDeltaFOverF(stack_2d[roi], None))

        out.append(np.vstack(out_curr))
        # print(out[-1].shape)
    # print(np.hstack(out).shape)
    savemat(output, {"deltafoverf":np.hstack(out)})

if __name__ == '__main__':
    fire.Fire(main)