import matplotlib.pyplot as plt
from dask import compute
from scipy.sparse import csr_matrix

import numpy as np

from scipy import io
import scipy.sparse as sparse
knn= 45
num_elements = 40000
normalize_w_k =30
def test_eigen():
    indices = io.loadmat('C:\\Users\gadge\Documents\CIDAN\inputs\\nf_0200_indices.mat')['Inds'].transpose((1,0))
    distances = io.loadmat('C:\\Users\gadge\Documents\CIDAN\inputs\\nf_0200_distances.mat')['Dis'].transpose((1,0))
    reformat_indicies_x = np.repeat(np.arange(0, num_elements, 1), knn - 1)
    reformat_indicies_y = np.reshape(indices[:, 1:], (-1))-1
    reformat_distances = np.reshape(distances[:, 1:], (-1))
    # Self tuning adaptive bandwidth
    scale_factor_indices = np.repeat(distances[:, normalize_w_k], knn)
    scale_factor_2_per_distances = np.power(scale_factor_indices[reformat_indicies_x],
                                            .5) * \
                                   np.power(scale_factor_indices[reformat_indicies_y],
                                            .5)
    scale_factor_2_per_distances[scale_factor_2_per_distances == 0] = 1
    reformat_distances_scaled = np.exp(
        -reformat_distances / scale_factor_2_per_distances)
    # TODO change to go direct to csr matrix
    K = sparse.csr_matrix(sparse.coo_matrix(
        (
            np.hstack([reformat_distances_scaled, np.ones((num_elements))]),
            (np.hstack([reformat_indicies_x, np.arange(0, num_elements, 1)]),
             np.hstack([reformat_indicies_y, np.arange(0, num_elements, 1)]))),
        shape=(num_elements, num_elements)))
    shape = (200, 200)
    num_points = 15
    shape_2d = (num_points * 25, 2)
    save_dir = "/Users/sschickler/Code Devel/LSSC-python/input_images/test2"
    image = np.zeros(shape_2d, dtype=np.float)
    for i in range(0, num_points, 1):
        for j in range(0, 25, 1):
            image[i * 25 + j, 0] = i
            image[i * 25 + j, 1] = j

    num_eig = 50
    # image_2d = reshape_to_2d_over_time(np.transpose(image,(2,0,1)))
    # K = calcAffinityMatrix(pixel_list=image, metric="l2", knn=20,
    #                        accuracy=80, connections=30,
    #                        normalize_w_k=15, num_threads=8, spatial_box_num=0,
    #                        temporal_box_num=0).compute()
    K_new = np.zeros((15 * 25, 15 * 25))
    for i in range(0, num_points, 1):
        for j in range(0, 25, 1):
            if (i > 0):
                K_new[i * 25 + j, (i - 1) * 25 + j] = 1
            if (i < num_points - 1):
                K_new[i * 25 + j, (i + 1) * 25 + j] = 1
            if (j > 0):
                K_new[i * 25 + j, (i) * 25 + j - 1] = 1
            if (j < 25 - 1):
                K_new[i * 25 + j, (i) * 25 + j + 1] = 1

    e_vectors = generateEigenVectors(K=csr_matrix(K_new), num_eig=num_eig)
    e_vectors = np.array(compute(e_vectors)[0])
    # plt.pcolormesh(e_vectors.transpose((1,0)))
    num = 0
    print(np.max(e_vectors[:, num]), np.min(e_vectors[:, num]))
    plt.scatter(image.transpose((1, 0))[0], image.transpose((1, 0))[1],
                c=e_vectors[:, num])
    plt.show()
    # save_image(image[:,:,0],0,shape,save_dir)
    # for x in range(num_eig):
    #     save_image(e_vectors[:, x], x + 1, shape, save_dir)


def save_image(image, num, shape, save_dir):
    e_vectors_sum_rescaled = image * (
            10.0 / image.max())  # add histogram equalization

    img = Image.fromarray(
        np.reshape(e_vectors_sum_rescaled,
                   shape) * 255).convert('L')
    image_path = os.path.join(save_dir, "eigen{}.png".format(
        num))
    img.save(image_path)


test_eigen()
