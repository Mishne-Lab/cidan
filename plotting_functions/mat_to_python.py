import json

import numpy as np
from scipy.io import loadmat

from cidan.LSSC.SpatialBox import SpatialBox

save_path = "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/KDA79_A_keep121.json"
x = loadmat(
    "/Users/sschickler/Code_Devel/LSSC-python/plotting_functions/KDA79_A_keep121.mat")
matrix = x['A_keep'].toarray()
roi_list = [np.flatnonzero(matrix[:, x]) for x in range(matrix.shape[1])]
spatial_box = SpatialBox(0, 1, image_shape=(128, 128), spatial_overlap=0)
roi_save_object = []
for num, roi in enumerate(roi_list):
    cords = spatial_box.convert_1d_to_2d(roi)
    cords = [(x[0] - 1, x[1] - 1) for x in cords]
    curr_roi = {"id": num, "coordinates": cords}
    roi_save_object.append(curr_roi)

with open(save_path, "w") as f:
    json.dump(roi_save_object, f)
