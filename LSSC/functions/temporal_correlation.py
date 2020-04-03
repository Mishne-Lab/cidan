import numpy as np
def calculate_temporal_correlation(dataset):
    shape = dataset.shape
    default_val = np.average(dataset)
    dataset_flat = dataset.transpose([1,2,0]).reshape([shape[1]*shape[2],-1])
    shift_up = np.hstack([np.full((shape[0],1, shape[2]),default_val),dataset[:,:-1,:]]).transpose([1,2,0]).reshape([shape[1]*shape[2],-1])
    shift_down = np.hstack([dataset[:,1:,:],np.full((shape[0],1, shape[2]), default_val)]).transpose([1,2,0]).reshape([shape[1]*shape[2],-1])
    shift_right = np.dstack([np.full((shape[0],shape[1], 1),default_val),dataset[:,:,:-1]]).transpose([1,2,0]).reshape([shape[1]*shape[2],-1])
    shift_left = np.dstack([np.full((shape[0],shape[1], 1),default_val),dataset[:,:,:-1]]).transpose([1,2,0]).reshape([shape[1]*shape[2],-1])
    corr_up = np.average(dataset_flat*shift_up,axis=1)
    corr_down = np.average(dataset_flat*shift_down,axis=1)
    corr_right = np.average(dataset_flat*shift_right,axis=1)
    corr_left = np.average(dataset_flat*shift_left,axis=1)
    avg_correlation = np.average(np.vstack([corr_up,corr_down,corr_left,corr_right]),axis=0)
    avg_correlation_image = avg_correlation.reshape([shape[1],shape[2]])
    return avg_correlation_image