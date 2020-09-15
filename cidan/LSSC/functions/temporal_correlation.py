import numpy as np
from dask import array


def calculate_temporal_correlation(dataset):
    shape = dataset.shape
    default_val = 0  # np.average(dataset)
    dataset_flat = array.from_array(
        dataset.transpose([1, 2, 0]).reshape([shape[1] * shape[2], -1]))

    corr_up = test_func(default_val, dataset_flat, shape, dataset).compute()
    shift_down = np.hstack(
        [dataset[:, 1:, :], np.full((shape[0], 1, shape[2]), default_val)]).transpose(
        [1, 2, 0]).reshape([shape[1] * shape[2], -1])
    corr_down = np.average(dataset_flat * shift_down, axis=1)
    del shift_down
    shift_right = np.dstack(
        [np.full((shape[0], shape[1], 1), default_val), dataset[:, :, :-1]]).transpose(
        [1, 2, 0]).reshape([shape[1] * shape[2], -1])
    corr_right = np.average(dataset_flat * shift_right, axis=1)
    del shift_right
    shift_left = np.dstack(
        [dataset[:, :, :-1], np.full((shape[0], shape[1], 1), default_val)]).transpose(
        [1, 2, 0]).reshape([shape[1] * shape[2], -1])
    corr_left = np.average(dataset_flat * shift_left, axis=1)
    del shift_left
    avg_correlation = np.average(np.vstack([corr_up, corr_down, corr_left, corr_right]),
                                 axis=0)
    avg_correlation_image = avg_correlation.reshape([shape[1], shape[2]])
    return avg_correlation_image


def test_func(default_val, dataset_flat, shape, dataset):
    shift_up = array.hstack(
        [array.zeros((shape[0], 1, shape[2])), dataset[:, :-1, :]]).transpose(
        [1, 2, 0]).reshape([shape[1] * shape[2], -1])

    shift_up_mult = dataset_flat * shift_up
    del shift_up
    return array.mean(shift_up_mult, axis=1)
