from ScanImageTiffReader import ScanImageTiffReader
import os
os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6
import scipy.sparse as sparse
import hnswlib as hnsw
from LSSC.Stack import *
from scipy.sparse import linalg
from PIL import Image

"""
I want to do a class system but take all the big logic outside the class into files of functions 
the data class will have functions to access all information and it will store all trial information in it 
If the info isn't avaiable the function will try to load it and then if that doesn't work generate it 
this will make the interactions really easy to write and 
"""
def load_tif_stack(path):
    """
    This function reads a tiff stack file
    Parameters
    ----------
    path The path to a single tif stack

    Returns
    -------
    a 3D numpy array with the tiff files together

    """
    if os.path.isdir(path):
        raise Exception( "Invalid Input folders not allowed currently ")
        # for num, x in enumerate(os.listdir(path)):
        #     file_path = os.path.join(path, x)
        #     if num == 0:
        #         vol = ScanImageTiffReader(file_path).data()
        #         print(vol)
    if os.path.isfile(path):
        return ScanImageTiffReader(path).data()[0::3, :, :]
    raise Exception("Invalid Input folders not allowed currently ")
    # vol=ScanImageTiffReader(file_path).data()

def reshape_to_2d_over_time(volume):
    """
    Takes a 3d numpy volume with dim 1 and 2 being x, y and dim 0 being time
    and return array compressing dim 1 and 2 into 1 dimension

    Parameters
    ----------
    volume input 3d numpy volume

    Returns
    -------
    a 2d volume with the 0 dim being pixel number and 1 being time value

    """
    return np.transpose(np.reshape(volume, (volume.shape[0],-1), order="C"))

def calc_affinity_matrix(pixel_list, metric="l2",knn=20,accuracy=200, connections=40, num_threads=10):
    """
    Calculates an affinity matrix for the image stack

    Parameters
    ----------
    pixel_list a 2d np array of pixels with dim 0 being a list of pixels and dim 1 being pixel values over time
    metric can be "l2" squared l2, "ip" Inner product, "cosine" Cosine similarity
    knn number of nearest neighbors to search for
    accuracy time of construction vs acuracy tradeoff
    connections max number of outgoing connections
    num_threads

    Returns
    -------

    """
    assert knn<accuracy, "Knn needs to be less than the accuracy ammount"
    # TODO make connections value scale based on available memory

    dim = pixel_list.shape[1]
    num_elements = pixel_list.shape[0]
    if pickle_exist("indices") and pickle_exist("distances"):
        indices, distances = pickle_load("indices"), pickle_load("distances")
    else:
        p = hnsw.Index(space='l2', dim=dim)
        p.init_index(max_elements=num_elements, ef_construction=accuracy, M=connections)
        p.add_items(pixel_list, num_threads=num_threads)
        indices, distances = p.knn_query(pixel_list, k=knn, num_threads=num_threads) #lazy random walk means it returns distance of zero for same point
        pickle_save(indices,"indices"), pickle_save(distances, "distances")

    reformat_indicies_x = np.repeat(np.arange(0, num_elements, 1), knn)
    reformat_indicies_y = np.reshape(indices, (-1))
    reformat_distances = np.reshape(distances, (-1))/np.median(np.reshape(distances, (-1))) #need to move this to the negative exponent of e talk about the bands
    std_indicies = np.std(distances, axis=1)
    std_2_per_distances = std_indicies[reformat_indicies_x]*std_indicies[reformat_indicies_y]
    reformat_distances_scaled = np.exp(-reformat_distances/std_2_per_distances)


    return sparse.csr_matrix(sparse.coo_matrix(
        (reformat_distances_scaled, (reformat_indicies_x, reformat_indicies_y)),
        shape=(num_elements, num_elements)))


def construct_D_inv(dim, K):
    """

    Parameters
    ----------
    dim dimensions for the graph
    K the sparse matrix K for the pairwise affinity

    Returns
    -------
    a sparse matrix with type csr
    """
    # D_diag = np.nan_to_num(1/K.sum(axis=1), nan=0.0, posinf =0, neginf=0) #add small epsilon to each row in K.sum()
    D_diag = 1/K.sum(axis=1)

    D_sparse = sparse.dia_matrix((np.reshape(D_diag, [1, -1]), [0]),
                      (dim, dim))
    return sparse.csr_matrix(D_sparse), D_diag
volume = load_tif_stack("/data2/Sam/pythonTestEnviroment/input_images/8_6_14_d10_001.tif")
# pickle_clear()
if pickle_exist("e_values") and pickle_exist("e_vectors"):
    e_values = pickle_load("e_values")
    e_vectors = pickle_load("e_vectors")
    D_neg_sqrt = pickle_load("D_neg_sqrt")

else:
    volume_reshape = reshape_to_2d_over_time(volume)
    K_sparse= calc_affinity_matrix(volume_reshape, knn=100, accuracy=400, connections=80,num_threads=10)
    num_elements = volume_reshape.shape[0]
    D_inv_sparse, D_diag = construct_D_inv(num_elements, K_sparse)
    I_sparse = sparse.identity(num_elements, format="csr")
    P_sparse = (D_inv_sparse * K_sparse)
    laplacian_sparse = I_sparse - P_sparse #don't use laplacian just use P
    D_sqrt = sparse.csr_matrix(sparse.dia_matrix((np.reshape(np.power(D_diag, .5), [1, -1]), [0]),
                               (num_elements, num_elements)))
    D_neg_sqrt = sparse.csr_matrix(
        sparse.dia_matrix((np.reshape(np.power(D_diag, -.5), [1, -1]), [0]),
                          (num_elements, num_elements)))

    P_transformed = D_sqrt*P_sparse*D_neg_sqrt
    e_values, e_vectors = linalg.eigsh(P_transformed, 200, which="LM", return_eigenvectors=True) #throw out first eigen vector of P because eigen vector is all ones and eigen value of 1
    pickle_save(e_values, "e_values")
    pickle_save(e_vectors, "e_vectors")
    pickle_save(D_neg_sqrt, "D_neg_sqrt")
e_vectors_squared = np.power((D_neg_sqrt*e_vectors), 2)
for x in range(200):
    img = Image.fromarray(
        np.reshape(e_vectors_squared[:,x]*(15.0/e_vectors_squared[:,x].max()), (512, 512)) * 255).convert('L')
    img.save('eigen_images/my%i.png'%x)
print(e_values)
e_vectors_sum = np.sum(e_vectors_squared, axis=1)
e_vectors_sum_rescaled = e_vectors_sum*(15.0/e_vectors_sum.max()) #add histogram equalization

img = Image.fromarray(np.reshape(e_vectors_sum_rescaled,(512,512))*255).convert('L')
img.save('my3.png')
img.show()
print(e_values)
print(e_vectors)




