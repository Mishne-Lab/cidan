import os
from functools import reduce
from typing import List, Tuple

from dask import delayed
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components as connected_components_graph
from skimage import measure
from skimage.feature import peak_local_max

from cidan.LSSC.functions.embeddings import embedEigenSqrdNorm
from cidan.LSSC.functions.progress_bar import printProgressBarROI
from cidan.LSSC.functions.widefield_functions import *


@delayed
def roi_extract_image(*, e_vectors: np.ndarray,
                      original_shape: tuple, original_2d_vol: np.ndarray, merge: bool,
                      num_rois: int, refinement: bool, num_eigen_vector_select: int,
                      max_iter: int, roi_size_min: int, fill_holes: bool,
                      elbow_threshold_method: bool, elbow_threshold_value: float,
                      eigen_threshold_method: bool,
                      eigen_threshold_value: float, merge_temporal_coef: float,
                      roi_size_limit: int, box_num: int, roi_eccentricity_limit: float,
                      total_num_spatial_boxes=0,
                      total_num_time_steps=0, save_dir=0, print_progress=False,
                      initial_pixel=-1, area_stop_threshold=0.95,

                      print_info=True, progress_signal=None, widefield=False,
                      image_data_mask=None,
                      local_max_method=False) -> List[
    np.ndarray]:
    """
    Computes the Local Selective Spectral roi_extraction algorithm on an set of
    eigen vectors
    Parameters
    ----------
    e_vectors
        Eigen vector in 2D numpy array
    original_shape
        Original shape of image
        Original
        shape of image
    original_2d_vol
        A flattened 2d volume of the original image, used for mergestep
    num_rois
        Number of rois
    refinement
        If to do roi refinement step
    num_eigen_vector_select
        Number of eigen values to project into
    max_iter
        Max amount of pixels to try and roi around
    roi_size_min
        Min size for roi to be output if too big might limit number of rois
    roi_size_limit
        max size for rois
    fill_holes
        fills holes in rois
    elbow_threshold_method
        whether to use elbow thresholding of the rois in refinement step
    elbow_threshold_value
        The value to use for elbow threshold
    eigen_threshold_method
        whether to use thresholding method when selecting eigen values
    eigen_threshold_value
        value for eigen thresholding method
    box_num
        Box number, just used for print statements
    merge
        whether to merge different rois based spatial and temporal information
    merge_temporal_coef
        The coefficient limiting merging based of temporal information, 0 merge all
        1 merge none
    initial_pixel
        used if you want to generate an roi from a specific pixel
    Returns
    -------
    2D list of rois [[np.array of pixels roi 1], [
                          np.array  of pixels roi 2] ... ]
    It will have length num_rois unless max_iter amount is surpassed
    """
    from cidan.LSSC.functions.data_manipulation import cord_2d_to_pixel_num

    # if print_info:
    #     print("Spatial Box {}: Starting ROI selection process".format(box_num))
    pixel_length = e_vectors.shape[0]
    if len(original_shape) == 2:
        original_shape = (1, original_shape[0], original_shape[1])

    pixel_embedings = embedEigenSqrdNorm(
        e_vectors)  # embeds the pixels in the eigen space

    if local_max_method or widefield:
        if widefield:
            pixel_embedings_all = mask_to_data_2d(pixel_embedings, image_data_mask)
        else:
            pixel_embedings_all = pixel_embedings
        image = np.reshape(pixel_embedings_all, original_shape[1:])
        image = gaussian_filter(image, np.std(image))
        local_max = peak_local_max(image, min_distance=int(
            roi_size_min * .25) if roi_size_min * .25 < 2 else 2)

        initial_pixel_list = cord_2d_to_pixel_num(local_max.transpose(),
                                                  original_shape[1:])
        sort_ind = np.flip(
            np.argsort(pixel_embedings_all[initial_pixel_list].flatten()))
        initial_pixel_list = initial_pixel_list[sort_ind]
        if widefield:
            initial_pixel_list = orig_to_mask_data_point(initial_pixel_list,
                                                         image_data_mask,
                                                         original_shape)
        print(len(initial_pixel_list))
        # if num_rois!=60:
        #     return [np.array([int(x)]) for x in initial_pixel_list]
        unassigned_pixels = np.arange(0, pixel_length, 1, dtype=int)
    else:
        initial_pixel_list = np.flip(np.argsort(
            pixel_embedings))  # creates a list of pixels with the highest values
    # in the eigenspace this list is used to decide on the initial point
    # for the roi
    """Plotting function, plots top 40 points:
    pixels = pixel_embedings.copy()
    pixels[initial_pixel_list[:40]]=100
    data=np.reshape(pixels, (400,150))
    plt.imshow(data)
    plt.show()"""
    if initial_pixel != -1:
        initial_pixel_list = np.array([initial_pixel])
    roi_list = []  # output list of rois
    # print(len(initial_pixel_list))
    # iter_counter is used to limit the amount of pixels it tries
    # from initial_pixel_list
    iter_counter = 0
    total_counter = 0
    pixels_assigned_set = {}
    while len(roi_list) < num_rois and len(
            initial_pixel_list) > 0 and iter_counter < max_iter and (
            not widefield or not float(
        len(pixels_assigned_set)) / pixel_length > area_stop_threshold or len(
        initial_pixel_list) != 0):

        # in this loop in widefield mode all pixel storage happens in masked state
        iter_counter += 1
        total_counter += 1
        # print(iter_counter, len(roi_list.json),
        #       len(roi_list.json[-1]) if len(roi_list.json) > 0 else 0)
        initial_pixel = initial_pixel_list[0]  # Select initial point
        # select eigen vectors to project into
        small_eigen_vectors = select_eigen_vectors(e_vectors,
                                                   [initial_pixel],
                                                   num_eigen_vector_select,
                                                   threshold_method=
                                                   eigen_threshold_method,
                                                   threshold=eigen_threshold_value)
        # print(small_eigen_vectors.shape)
        # print("original",smam nm ll_eigen_vectors.shape)
        # TODO Find way to auto determine threshold value automatically max values
        # project into new eigen space
        small_pixel_embeding_norm = embedEigenSqrdNorm(small_eigen_vectors)

        # calculate the distance between the initial point and each pixel
        # in the new eigen space
        small_pixel_distance = pixel_distance(small_eigen_vectors,
                                              initial_pixel)

        # selects pixels in roi
        pixels_in_roi = np.nonzero(
            small_pixel_distance <= small_pixel_embeding_norm)[0]
        if widefield:
            # runs a connected component analysis around the initial point
            # in original image
            pixels_in_roi_comp = connected_component(pixel_length,
                                                     original_shape,
                                                     mask_to_data_point(
                                                         pixels_in_roi, image_data_mask,
                                                         original_shape),
                                                     mask_to_data_point(
                                                         [initial_pixel],
                                                         image_data_mask,
                                                         original_shape), )

            pixels_in_roi_final = orig_to_mask_data_point(pixels_in_roi_comp,
                                                          image_data_mask,
                                                          original_shape)
        else:
            # runs a connected component analysis around the initial point
            # in original image
            pixels_in_roi_comp = connected_component(pixel_length,
                                                     original_shape,
                                                     pixels_in_roi,
                                                     initial_pixel)

            pixels_in_roi_final = pixels_in_roi_comp
        # runs refinement step if enabled and if enough pixels in roi
        if refinement:  # TODO Look at this again
            # and len(pixels_in_roi_final) > \
            #  roi_size_threshold / 2
            # selects a new set of eigenvectors based on the pixels in roi
            rf_eigen_vectors = select_eigen_vectors(e_vectors,
                                                    pixels_in_roi_final,
                                                    num_eigen_vector_select,
                                                    threshold_method=eigen_threshold_method,
                                                    threshold=eigen_threshold_value)

            # print("rf",rf_eigen_vectors.shape)
            # embeds all pixels in this new eigen space
            rf_pixel_embedding_norm = embedEigenSqrdNorm(rf_eigen_vectors)

            # selects the initial point based on the pixel with max in
            # the new embedding space
            rf_initial_point = rf_select_initial_point(rf_pixel_embedding_norm,
                                                       pixels_in_roi_final)

            # calculate the distance between the initial point and each pixel
            # in the new eigen space
            rf_pixel_distance = pixel_distance(rf_eigen_vectors,
                                               rf_initial_point)

            # selects pixels in roi
            if elbow_threshold_method:
                threshold = elbow_threshold_value * elbow_threshold(rf_pixel_distance,
                                                                    np.argsort(
                                                                        rf_pixel_distance),
                                                                    half=True)
                rf_pixels_in_roi = np.nonzero(
                    rf_pixel_distance < threshold)[0]
            else:
                rf_pixels_in_roi = np.nonzero(
                    rf_pixel_distance <= rf_pixel_embedding_norm)[0]

            # runs a connected component analysis around the initial point
            # in original image
            if widefield:
                rf_pixels_in_roi_comp = connected_component(pixel_length,
                                                            original_shape,
                                                            mask_to_data_point(
                                                                rf_pixels_in_roi,
                                                                image_data_mask,
                                                                original_shape),
                                                            mask_to_data_point(
                                                                [rf_initial_point],
                                                                image_data_mask,
                                                                original_shape))
                rf_pixels_in_roi_filled = \
                    fill_holes_func([rf_pixels_in_roi_comp], pixel_length,
                                    original_shape)[
                        0]  # TODO do we fill the rois in widefield
                pixels_in_roi_final = orig_to_mask_data_point(rf_pixels_in_roi_filled,
                                                              image_data_mask,
                                                              original_shape)

            else:
                rf_pixels_in_roi_comp = connected_component(pixel_length,
                                                            original_shape,
                                                            rf_pixels_in_roi,
                                                            rf_initial_point)
                rf_pixels_in_roi_filled = \
                    fill_holes_func([rf_pixels_in_roi_comp], pixel_length,
                                    original_shape)[
                        0]
                pixels_in_roi_final = rf_pixels_in_roi_filled

        # checks if roi is big enough
        # print("roi size:", len(pixels_in_roi_final))
        # print("iter counter: ", iter_counter)
        # print( len(
        #         pixels_in_roi_final))

        if roi_size_min < len(
                pixels_in_roi_final) < roi_size_limit and (
                widefield or roi_eccentricity(pixel_length,
                                              original_shape,
                                              pixels_in_roi_final) <= roi_eccentricity_limit):

            roi_list.append(pixels_in_roi_final)

            iter_counter = 0
            if widefield:
                initial_pixel_list = np.delete(initial_pixel_list, 0)

            else:

                # takes all pixels in current roi out of initial_pixel_list
                initial_pixel_list = np.extract(
                    np.in1d(initial_pixel_list, pixels_in_roi_final,
                            assume_unique=True, invert=True),
                    initial_pixel_list)
                if initial_pixel not in pixels_in_roi_final:
                    initial_pixel_list = np.delete(initial_pixel_list, 0)

            # print(len(initial_pixel_list))
        else:
            if widefield:
                initial_pixel_list = np.delete(initial_pixel_list, 0)
            else:
                # takes current initial point and moves it to end of
                # initial_pixel_list
                initial_pixel_list = np.delete(
                    np.append(initial_pixel_list, initial_pixel_list[0]), 0)
    if widefield:
        roi_list = [mask_to_data_point(x, image_data_mask, original_shape) for x in
                    roi_list]  # converts all points to unmasked version
    #### in widefield all points are now in unmasked space
    if fill_holes:
        # TODO combine into connected component function
        roi_list = fill_holes_func(roi_list, pixel_length, original_shape)
    # Merges rois
    if merge:
        roi_list = merge_rois(roi_list,
                              temporal_coefficient=merge_temporal_coef,
                              original_2d_vol=original_2d_vol,
                              roi_eccentricity_limit=roi_eccentricity_limit,
                              widefield=widefield)
        if fill_holes:
            roi_list = fill_holes_func(roi_list, pixel_length, original_shape)
    if widefield:
        # handles overlapping rois
        roi_list = remove_overlap_widefield(roi_list, image_data_mask, original_shape,
                                            e_vectors)
        roi_list = add_unassigned_pixels_widefield(roi_list, image_data_mask,
                                                   original_shape, e_vectors)
    # new_rois_filtered= []
    # for roi in roi_list:
    #     if roi_eccentricity(pixel_length,original_shape,roi)<=roi_eccentricity_limit:
    #         new_rois_filtered.append(roi)
    # roi_list=new_rois_filtered
    # print("Went through " + str(total_counter) + " iterations")
    if print_progress:
        with open(os.path.join(save_dir, "temp_files/rois/s_%s" % str(box_num)),
                  "w") as f:
            f.write("done")
        printProgressBarROI(total_num_spatial_boxes, total_num_time_steps, save_dir,
                            progress_signal=progress_signal)
    roi_list = [np.array(x) for x in roi_list]
    return roi_list


def add_unassigned_pixels_widefield(roi_list, mask, original_shape, e_vectors):
    new_rois = []
    for roi in roi_list:
        image_temp = np.zeros((original_shape[1] * original_shape[2]), dtype=float)

        image_temp[roi] = 1
        image_temp = image_temp.reshape((original_shape[1], original_shape[2]))
        # edge = feature.canny(
        #     np.sum(image_temp, axis=2) / np.max(np.sum(image_temp, axis=2)))
        # image[edge] = 1
        image_temp = ndimage.morphology.binary_dilation(image_temp)
        image_temp = image_temp.reshape((original_shape[1] * original_shape[2]))
        new_rois.append(np.nonzero(image_temp))
    roi_pixel_matrix = np.zeros((len(roi_list), original_shape[1] * original_shape[2]))
    pixels_currently_in_roi = np.hstack(roi_list)
    for num, roi in enumerate(new_rois):
        roi_pixel_matrix[num, roi] = 1
    roi_pixel_matrix[:, pixels_currently_in_roi] = 0
    roi_pixel_matrix[:, ~mask.flatten()] = 0
    num_rois_pixels = np.sum(roi_pixel_matrix, axis=0)
    pixels_to_assign = np.nonzero(num_rois_pixels == 1)[0]

    for pixel in pixels_to_assign:
        roi_num = np.nonzero(roi_pixel_matrix[:, pixel])[0][0]
        roi_list[roi_num] = np.append(roi_list[roi_num], pixel)
    roi_centroids = []

    def change_1_cord(cord_1d):
        # converts 1d cord to 2d cord
        return int(cord_1d // original_shape[1]), int(cord_1d - (
                cord_1d // original_shape[1]) * original_shape[1])

    for roi in roi_list:
        roi_centroids.append(
            np.mean(e_vectors[orig_to_mask_data_point(roi, mask, original_shape)],
                    axis=0))
    pixels_with_two_overlap = np.nonzero(num_rois_pixels >= 2)[0]
    for pixel in pixels_with_two_overlap:
        rois = np.nonzero(roi_pixel_matrix[:, pixel])[0]
        diffs = []
        for roi_num in rois:
            diffs.append(
                distance_from_centroid(roi_list[roi_num], [pixel], mask, original_shape,
                                       e_vectors, roi_centroids[roi_num]))
        closest_roi = rois[np.argmin(diffs)]
        roi_list[closest_roi] = np.append(roi_list[closest_roi], pixel)

    return roi_list


def distance_from_centroid(roi, points, mask, original_shape, e_vectors,
                           centroid_vector):
    """
    Calculates the distance of a set of points from the rois centroid
    Parameters
    ----------
    roi list of points in original shape cords 1d
    points list of points in original shape cords 1d
    mask
    original_shape
    e_vectors

    Returns
    -------

    """

    embedding_centroid = centroid_vector

    embedding_points = e_vectors[orig_to_mask_data_point(points, mask, original_shape)]
    diff = embedEigenSqrdNorm(embedding_points - embedding_centroid)
    return diff
def remove_overlap_widefield(roi_list, mask, original_shape, e_vectors):
    """
    Removes overlapping pixels in widefield.
    Parameters
    ----------
    roi_list
    mask
    original_shape
    e_vectors

    Returns
    -------

    """
    # [number_connected_components(pixel_length=original_shape[1]*original_shape[2], original_shape=original_shape,pixels_in_roi=x)for x in roi_list]
    A = np.zeros([original_shape[1] * original_shape[2], len(roi_list)], dtype=bool)
    for num, roi in enumerate(roi_list):
        A[roi, num] = True
    A_graph = np.matmul(A.transpose(), A)
    overlaping_rois = np.nonzero(np.triu(A_graph))

    roi_centroids = []

    def change_1_cord(cord_1d):
        # converts 1d cord to 2d cord
        return int(cord_1d // original_shape[1]), int(cord_1d - (
                cord_1d // original_shape[1]) * original_shape[1])

    for roi in roi_list:
        roi_centroids.append(
            np.mean(e_vectors[orig_to_mask_data_point(roi, mask, original_shape)],
                    axis=0))
    for roi_a, roi_b in list(zip(overlaping_rois[0], overlaping_rois[1])):
        if roi_a != roi_b:
            overlap_pixels = np.intersect1d(roi_list[roi_a], roi_list[roi_b],
                                            assume_unique=True)
            diff_a = distance_from_centroid(roi_list[roi_a], overlap_pixels, mask,
                                            original_shape, e_vectors,
                                            centroid_vector=roi_centroids[roi_a])
            diff_b = distance_from_centroid(roi_list[roi_b], overlap_pixels, mask,
                                            original_shape, e_vectors,
                                            centroid_vector=roi_centroids[roi_b])
            remove_from_a = overlap_pixels[diff_a > diff_b]
            remove_from_b = overlap_pixels[diff_a <= diff_b]
            roi_list[roi_a] = np.setdiff1d(roi_list[roi_a], remove_from_a,
                                           assume_unique=True)
            roi_list[roi_b] = np.setdiff1d(roi_list[roi_b], remove_from_b,
                                           assume_unique=True)
    return roi_list


def fill_holes_func(roi_list: List[np.ndarray], pixel_length: int,
                    original_shape: Tuple[int, int, int]) -> List[np.ndarray]:
    """
    Close holes in each roi
    Parameters
    ----------
    roi_list
        List of Rois in format: [[np.array of pixels roi 1],
        [np.array  of pixels roi 2] ... ]
    pixel_length
        Number of pixels in image
    original_shape
        Original shape of the image

    Returns
    -------
    roi list with holes filled in format: [[np.array of pixels roi 1],
        [np.array  of pixels roi 2] ... ]
    """
    for num, roi in enumerate(roi_list):
        original_zeros = np.zeros((original_shape[1] * original_shape[2]))
        original_zeros[roi] = 255
        image_filled = binary_fill_holes(np.reshape(original_zeros,
                                                    original_shape[1:]))
        image_filled_2d = np.reshape(image_filled, (-1))
        roi_list[num] = np.nonzero(image_filled_2d)[0]
    return roi_list


def pixel_distance(eigen_vectors: np.ndarray, pixel_num: int) -> np.ndarray:
    """
    Calculates squared distance between pixels in embedding space and initial_point
    Parameters
    ----------
    eigen_vectors
        The eigen vectors describing the vector space with
        dimensions number of pixels in image by number of eigen vectors
    pixel_num
        The number of the initial pixel in the eigen vectors

    Returns
    -------
    A np array with dim: number of pixels in image
    """
    return np.sum(np.power(eigen_vectors - eigen_vectors[pixel_num], 2),
                  axis=1)


def number_connected_components(pixel_length: int, original_shape: Tuple[int, int, int],
                                pixels_in_roi: np.ndarray):
    """
        Runs a connected component analysis on a group of pixels in an image
        Parameters
        ----------
        pixel_length
            Number of pixels in image
        original_shape
            the original shape of image
        pixels_in_roi
            A list of pixels in the roi


        Returns
        -------
        Number of groups of pixels in the image
    """
    original_zeros = np.zeros(pixel_length)
    original_zeros[pixels_in_roi] = 1
    pixel_image = np.reshape(original_zeros, original_shape[1:])
    # runs connected component analysis on image
    blobs_labels = np.reshape(measure.label(pixel_image, background=0),
                              (-1))
    return np.unique(blobs_labels).shape[0]
def connected_component(pixel_length: int, original_shape: Tuple[int, int, int],
                        pixels_in_roi: np.ndarray,
                        initial_pixel_number: int) -> np.ndarray:
    """
    Runs a connected component analysis on a group of pixels in an image
    Parameters
    ----------
    pixel_length
        Number of pixels in image
    original_shape
        the original shape of image
    pixels_in_roi
        A list of pixels in the roi
    initial_pixel_number
        The number of the original pixel in the
        flattened image


    Returns
    -------
    An subset of the original pixels that are connected to initial pixel
    """
    # TODO add in im fill before connected component
    # first creates an image with pixel values of 1 if pixel in roi
    original_zeros = np.zeros(original_shape[1] * original_shape[2])
    original_zeros[pixels_in_roi] = 1
    pixel_image = np.reshape(original_zeros, original_shape[1:])
    # runs connected component analysis on image
    blobs_labels = np.reshape(measure.label(pixel_image, background=0),
                              (-1))
    correct_label = blobs_labels[initial_pixel_number]

    # filters pixels to only ones with same label as initial pixel
    pixels_in_roi_new = np.nonzero(
        blobs_labels == correct_label)[0]
    return pixels_in_roi_new
def roi_eccentricity(pixel_length: int, original_shape: Tuple[int, int, int],
                        pixels_in_roi: np.ndarray) -> np.ndarray:
    """
    Runs a eccentricity analysis on a group of pixels in an image
    Parameters
    ----------
    pixel_length
        Number of pixels in image
    original_shape
        the original shape of image
    pixels_in_roi
        A list of pixels in the roi



    Returns
    -------
    Eccentricity of roi, with 0 being circle 1 being a line
    """
    # TODO add in im fill before connected component
    # first creates an image with pixel values of 1 if pixel in roi
    original_zeros = np.zeros(pixel_length, dtype=int)
    original_zeros[pixels_in_roi] = 1
    pixel_image = np.reshape(original_zeros, original_shape[1:])
    eccentricity = measure.regionprops(pixel_image)[0]["eccentricity"]
    return eccentricity


def select_eigen_vectors(eigen_vectors: np.ndarray,
                         pixels_in_roi: np.ndarray,
                         num_eigen_vector_select: int,
                         threshold_method: bool = False,
                         threshold: float = .9) -> np.ndarray:
    """
    Selects eigen vectors that are most descriptive of a set a points
    Parameters
    ----------
    eigen_vectors
        The eigen vectors describing the vector space with
        dimensions number of pixels in image by number of eigen vectors
    pixels_in_roi
        Np array of indices of all pixels in roi
    num_eigen_vector_select
        Number of eigen vectors to select
    threshold_method
        this is a bool on whether to run the threshold method to select the eigen
        vectors threshold

    Returns
    -------
    the eigen vectors describing the new vector space with
        dimensions number of pixels in image by numb_eigen_vector_select

    """
    # pixel_eigen_vec_values = np.abs(np.sum(eigen_vectors[pixels_in_roi], axis=0))
    pixel_eigen_vec_values = np.power(np.sum(eigen_vectors[pixels_in_roi], axis=0), 2)

    pixel_eigen_vec_values_sort_indices = np.flip(
        np.argsort(pixel_eigen_vec_values))
    if threshold_method:
        threshold_filter = pixel_eigen_vec_values > (1 - threshold) * \
                           pixel_eigen_vec_values[
                               pixel_eigen_vec_values_sort_indices[0]]
        small_eigen_vectors = eigen_vectors[:, np.nonzero(threshold_filter)[0]]

    if not threshold_method or small_eigen_vectors.shape[1] < num_eigen_vector_select:
        pixel_eigen_vec_values_sort_indices = np.flip(
            np.argsort(
                pixel_eigen_vec_values))
        small_eigen_vectors = eigen_vectors[:,
                              pixel_eigen_vec_values_sort_indices[
                              :num_eigen_vector_select]]
    return small_eigen_vectors


def rf_select_initial_point(pixel_embedings: np.ndarray,
                            pixels_in_roi: np.ndarray):
    """
    Selects an initial point for roi_extraction based on the pixels in current
    roi, this is part of the refinement step
    Parameters
    ----------
    pixel_embedings
        The embedings of each pixel in a vector space
    pixels_in_roi
        a list of the indices of the pixel in the roi

    Returns
    -------
    an indice for the initial point pixel
    """
    indice_in_roi = \
        np.flip(np.argsort(pixel_embedings[pixels_in_roi]))[0]
    return np.sort(pixels_in_roi)[indice_in_roi]


def elbow_threshold(pixel_vals: np.ndarray, pixel_val_sort_indices: np.ndarray,
                    half: bool = True) -> float:
    """
    Calculates the elbow threshold for the refinement step in roi_extraction algorithm,


    It determines the pixel that is farthest away from line drawn from the line from
    first to last or first to middle pixel. To find pixel it then projects each
    point(pixel # in sorted pixel_val, distance val) to the line then subtracts that
    from the points value to find distance from line

    Parameters
    ----------
    pixel_vals
        The distance values for each pixel
    pixel_val_sort_indices
        The array necessary to sort said pixels from lowest to highest
    half
        whether to run only with the closest half of the pixels, recommended

    Returns
    -------
    float, the optimal threshold based on elbow
    """
    n_points = len(pixel_vals) if not half else len(pixel_vals) // 2

    pixel_vals_sorted_zipped = np.array(list(
        zip(range(n_points), pixel_vals[pixel_val_sort_indices[:n_points]])))

    first_point = pixel_vals_sorted_zipped[0, :]
    last_point = pixel_vals_sorted_zipped[-1, :]
    line_vec = last_point - first_point

    line_vec_norm = line_vec / (np.sum(np.power(line_vec, 2)) ** .5)
    dist_from_first = pixel_vals_sorted_zipped - first_point
    proj_point_to_line = np.matmul(dist_from_first,
                                   line_vec_norm[:, None]) * line_vec_norm
    vec_to_line = dist_from_first - proj_point_to_line
    dist_to_line = np.power(np.sum(np.power(vec_to_line, 2), axis=1), .5)
    # fig, ax = plt.subplots(figsize=(4, 4))
    # ax.scatter(list(range(n_points)),
    #            dist_from_first[:n_points,1])
    # fig.savefig(
    #     "/Users/sschickler/Documents/LSSC-python/output_images/plots/dist_plot_{}2.png".format(num),
    #     aspect='auto')
    dist_max_indice = np.argmax(dist_to_line)
    threshold = pixel_vals_sorted_zipped[dist_max_indice][1]
    return threshold


def merge_rois(roi_list: List,
               temporal_coefficient: float, original_2d_vol: np.ndarray,
               roi_eccentricity_limit=1.0, widefield=False):
    # TODO is this the most efficient implementation I can do
    """
    Merges rois based on temporal and spacial overlap
    Parameters
    ----------
    roi_list
        List of Rois in format: [[np.array of pixels roi 1],
        [np.array  of pixels roi 2] ... ]
    temporal_coefficient
        The coefficient limiting merging based of temporal information, 0 merge all
        1 merge none
    original_2d_vol
        Volume of each pixel's time trace
    Returns
    -------
        List of new rois in format: [[np.array of pixels roi 1],
        [np.array  of pixels roi 2] ... ]
    """
    A = np.zeros([original_2d_vol.shape[0], len(roi_list)], dtype=int)  # create 2d
    # matrix of zeros with dims number of pixels in image by number of rois
    # Change pixels of each roi to 1
    for num, roi in enumerate(roi_list):
        A[roi, num] = 1
    # Create graph of which rois have pixels which intersect with each other.
    A_graph = np.matmul(A.transpose(), A)
    connected_rois = np.nonzero(A_graph)
    # print(A_graph)
    timetraces = [np.mean(original_2d_vol[roi], axis=0) for roi in roi_list]
    A_graph_new = np.identity(A_graph.shape[0], dtype=float)
    # print(list(zip(*connected_rois)))
    for x in list(zip(*connected_rois)):
        # applies a 10% overlap condition to the rois.
        if x[0] != x[1] and (widefield or (
                A_graph[x[0], x[1]] > len(roi_list[x[1]]) * .1 and A_graph[
            x[0], x[1]] > len(roi_list[x[0]]) * .1)):
            A_graph_new[x[0], x[1]] = compare_time_traces(timetraces[x[0]],
                                                          timetraces[x[1]])
            # print(A_graph_new[x[0],x[1]])
            A_graph_new[x[1], x[0]] = A_graph_new[x[0], x[1]]
            A_graph[x[0], x[1]] = False
            A_graph[x[1], x[0]] = False

    A_components_to_merge = A_graph_new >= temporal_coefficient
    A_csr = csr_matrix(A_components_to_merge)
    # Use connected components to group these rois together
    connected = connected_components_graph(A_csr, False, return_labels=True)
    # processes connected components putting each group of rois into roi_groups list
    roi_groups = [[] for _ in range(len(roi_list))]
    for num in range(len(roi_list)):
        roi_groups[connected[1][num]].append(roi_list[num])

    new_rois = []
    for group in roi_groups:
        if len(group) != 0:
            # combine those rois that should be merged with first roi.
            first_roi = list(reduce(combine_rois, group))

            new_rois.append(np.array(first_roi))

    return new_rois


def compare_time_traces(trace_1: np.ndarray, trace_2: np.ndarray) -> float:
    """
    Compares two time traces based on pearson correlation

    Parameters
    ----------
    trace_1
        A 2d numpy list of values where 1 dimensions is each pixel in roi 1 and other is
        a time trace for each pixel
    trace_2
        A 2d numpy list of values where 1 dimensions is each pixel in roi 2 and other is
        a time trace for each pixel

    Returns
    -------
    the correlation as a float
    """
    trace_1_mean = np.mean(trace_1)
    trace_2_mean = np.mean(trace_2)
    trace_1_sub_mean = (trace_1 - trace_1_mean)
    trace_2_sub_mean = (trace_2 - trace_2_mean)
    top = np.dot(trace_1_sub_mean, trace_2_sub_mean)
    bottom = (np.dot(trace_1_sub_mean, trace_1_sub_mean) ** .5 *
              np.dot(trace_2_sub_mean, trace_2_sub_mean) ** .5)
    return top / bottom


def combine_rois(roi1: List[int], roi2: List[int]) -> List[np.ndarray]:
    """
    Combines two lists of rois into one
    Parameters
    ----------
    roi1
        One list of pixels in roi each pixel # is based on 1d representation of
        image
    roi2
        List for other ROI

    Returns
    -------
    List of merged ROI
    """
    roi2_not_in_1 = list(filter(lambda x: x not in roi1, roi2))
    return np.array(list(roi1) + roi2_not_in_1)
