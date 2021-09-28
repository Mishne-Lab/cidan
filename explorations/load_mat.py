import h5py
import numpy as np


def load_dataset_widefield_test():
    file = "/Users/sschickler/Code_Devel/Widefield/widefield_firdff.mat"
    with h5py.File(file, 'r') as f:
        data_all = np.array(f["dff_final"][:500, :]).astype(np.float64).transpose()
        mask = np.array(f["mask"]).astype(bool)
    data_mask = np.isnan(data_all)
    data_mask[~mask.flatten()] = False
    nan_means = np.nanmean(data_all, axis=1)

    data_all = np.where(np.isnan(data_all),
                        np.repeat(np.nanmean(data_all, axis=1).reshape((-1, 1)),
                                  data_all.shape[1], axis=1), data_all)
    # data_all = np.nan_to_num(data_all)
    # imwrite("test2.tif",data_all.reshape((mask.shape[0], mask.shape[1], data_all.shape[1])).transpose((2,0,1)), photometric='minisblack')
    # im.save('/Users/sschickler/Code_Devel/LSSC-python/explorations/test2.tif')
    return data_all, mask


if __name__ == '__main__':
    load_dataset_widefield_test()
