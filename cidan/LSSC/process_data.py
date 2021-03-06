import logging
import os
from functools import reduce
from typing import List

import dask
import numpy as np
from dask import delayed

from cidan.LSSC.SpatialBox import SpatialBox
from cidan.LSSC.functions.data_manipulation import reshape_to_2d_over_time, \
    join_data_list
from cidan.LSSC.functions.eigen import generateEigenVectors, saveEigenVectors, \
    loadEigenVectors, saveEmbedingNormImage, createEmbedingNormImageFromMultiple
from cidan.LSSC.functions.embeddings import calcAffinityMatrix
from cidan.LSSC.functions.progress_bar import printProgressBarROI
from cidan.LSSC.functions.roi_extraction import roi_extract_image, merge_rois
from cidan.LSSC.functions.save_test_images import save_roi_images

logger1 = logging.getLogger("cidan.LSSC.process_data")


def process_data(*, test_images: bool, test_output_dir: str,
                 save_dir: str, save_intermediate_steps: bool,
                image_data: List[np.ndarray],
                 image_data_filtered: List[np.ndarray],
                 shape: List[np.ndarray],
                 crop: List[np.ndarray],
                 slicing: List[np.ndarray],
                 eigen_vectors_already_generated: bool,
                 save_embedding_images: bool,
                 total_num_time_steps: int, total_num_spatial_boxes: int,
                 spatial_overlap: int, metric: str, knn: int,
                 accuracy: int, connections: int, normalize_w_k: int, num_eig=25,
                 merge: bool,
                 num_rois: int, refinement: bool, num_eigen_vector_select: int,
                 max_iter: int, roi_size_min: int, fill_holes: bool,
                 elbow_threshold_method: bool, elbow_threshold_value: float,
                 eigen_threshold_method: bool,
                 eigen_threshold_value: float, merge_temporal_coef: float,
                 roi_size_max: int, pca: bool, pca_data: np.ndarray,
                 eigen_accuracy: int, roi_eccentricity_limit: float,
                 local_max_method: bool, progress_signal=None):
    logger1.debug("""Inputs: num_threads {0},test_images {1}, test_output_dir {2},
                 save_dir {3}, save_intermediate_steps {4},
                 load_data {5}, data_path {6},
                 image_data {7},
                 eigen_vectors_already_generated {8},
                 save_embedding_images {9},
                 total_num_time_steps {10}, total_num_spatial_boxes {11},
                 spatial_overlap {12}, filter {13}, median_filter {14},
                 median_filter_size {15},
                 z_score {16}, slice_stack {17},
                 slice_every {18}, slice_start {19}, metric {20}, knn {21},
                 accuracy {22}, connections {23}, normalize_w_k {24}, num_eig {25},
                 merge {26},
                 num_rois {27}, refinement {28}, num_eigen_vector_select {29},
                 max_iter {30}, roi_size_min {31}, fill_holes {32},
                 elbow_threshold_method {33}, elbow_threshold_value {34},
                 eigen_threshold_method {35},
                 eigen_threshold_value {36}, merge_temporal_coef {37},
                 roi_size_max {38}""".format("none", test_images, test_output_dir,
                                             save_dir, save_intermediate_steps,
                                             "", "",
                                             image_data_filtered,
                                             eigen_vectors_already_generated,
                                             save_embedding_images,
                                             total_num_time_steps,
                                             total_num_spatial_boxes,
                                             spatial_overlap, "", "", "", "", "", "",
                                             "",
                                             "", metric, knn,
                                             accuracy, connections, normalize_w_k,
                                             num_eig,
                                             merge,
                                             num_rois, refinement,
                                             num_eigen_vector_select,
                                             max_iter, roi_size_min, fill_holes,
                                             elbow_threshold_method,
                                             elbow_threshold_value,
                                             eigen_threshold_method,
                                             eigen_threshold_value, merge_temporal_coef,
                                             roi_size_max))
    # TODO Make after eigen vector make function to save intermediate embeding norm
    #  for each spatial box
    # TODO Rewrite to take in a list of loaded datasets

    # TODO add assertions to make sure input splits work for dataset
    if save_intermediate_steps and not eigen_vectors_already_generated:
        filelist = [f for f in os.listdir(os.path.join(save_dir, "eigen_vectors"))]
        for f in filelist:
            os.remove(os.path.join(os.path.join(save_dir, "eigen_vectors"), f))

    if not os.path.isdir(os.path.join(save_dir, "temp_files/embedding")):
        os.mkdir(os.path.join(save_dir, "temp_files/embedding"))
    if not os.path.isdir(os.path.join(save_dir, "temp_files/rois")):
        os.mkdir(os.path.join(save_dir, "temp_files/rois"))
    filelist = [f for f in os.listdir(os.path.join(save_dir, "temp_files/embedding"))]
    for f in filelist:
        os.remove(os.path.join(os.path.join(save_dir, "temp_files/embedding"), f))
    filelist = [f for f in os.listdir(os.path.join(save_dir, "temp_files/rois"))]
    for f in filelist:
        os.remove(os.path.join(os.path.join(save_dir, "temp_files/rois"), f))
    # shape = [image_data_filtered[0].shape[1], image_data_filtered[0].shape[2]]
    logger1.debug("image shape {0}".format(shape))
    # print("Creating {} spatial boxes".format(total_num_spatial_boxes))
    printProgressBarROI(total_num_spatial_boxes=total_num_spatial_boxes,
                        total_num_time_steps=total_num_time_steps,
                        save_dir=save_dir, progress_signal=progress_signal)
    spatial_boxes = [SpatialBox(box_num=x, total_boxes=total_num_spatial_boxes,
                                spatial_overlap=spatial_overlap, image_shape=shape)
                     for x in range(total_num_spatial_boxes)]
    all_rois = []
    all_boxes_eigen_vectors = []
    for spatial_box in spatial_boxes:
        spatial_box_data_list = [convert_to_float(spatial_box.extract_box(x)) for x in
                                 image_data_filtered]
        if pca:
            spatial_box_data_list_pca = [spatial_box.extract_box(x) for x in pca_data]
        if len(spatial_box_data_list) == 1:
            spatial_box_data_list = dask.compute(*spatial_box_data_list)
            if pca:
                spatial_box_data_list_pca = dask.compute(*spatial_box_data_list_pca)


        if total_num_time_steps != 1 and len(spatial_box_data_list) == 1 and not pca:
            time_boxes = [(x * (image_data_filtered[0].shape[0] // total_num_time_steps),
                           (x + 1) * (image_data_filtered[0].shape[0] //
                                      total_num_time_steps))
                          for x in range(total_num_time_steps)]


        all_eigen_vectors_list = []
        if not eigen_vectors_already_generated:
            for temporal_box_num, time_box_data in enumerate(
                    time_boxes if total_num_time_steps != 1 and len(
                        spatial_box_data_list) == 1 else spatial_box_data_list):
                # TODO make sure memory doesn't take more than 2x
                if total_num_time_steps != 1 and len(
                        spatial_box_data_list) == 1 and not pca:
                    time_box_data = sliceData(spatial_box_data_list,
                                              start_end=time_box_data)
                # time_box_data = saveTempImage(time_box_data, save_dir,
                #                               spatial_box.box_num)

                time_box_data_2d = reshape_to_2d_over_time(time_box_data)
                if pca:
                    time_box_data_2d_pca = reshape_to_2d_over_time(
                        spatial_box_data_list_pca[temporal_box_num])
                logger1.debug(
                    "Time box {0}, start {1}, end {2}, time_box shape {3}, 2d shape {4}".format(
                        temporal_box_num, 0, 0, time_box_data.shape,
                        time_box_data_2d.shape))
                k = calcAffinityMatrix(
                    pixel_list=time_box_data_2d if not pca else time_box_data_2d_pca,
                    metric=metric,
                    knn=knn, accuracy=accuracy,
                    connections=connections,
                    normalize_w_k=25,
                    num_threads=8,
                    spatial_box_num=spatial_box.box_num,
                    temporal_box_num=temporal_box_num,
                    total_num_spatial_boxes=total_num_spatial_boxes,
                    total_num_time_steps=total_num_time_steps, save_dir=save_dir,
                    progress_signal=progress_signal)
                eigen_vectors = generateEigenVectors(K=k,
                                                     num_eig=time_box_data_2d_pca.shape[
                                                         1] if pca and False and num_eig >
                                                               pca_data[temporal_box_num].shape[
                                                                   0] else num_eig,
                                                     accuracy=10 ** (
                                                                 -1 * eigen_accuracy)

                                                     )
                if save_intermediate_steps:
                    eigen_vectors = saveEigenVectors(e_vectors=eigen_vectors,
                                                     spatial_box_num=spatial_box.box_num,
                                                     time_box_num=temporal_box_num,
                                                     save_dir=save_dir,
                                                     total=total_num_time_steps * total_num_spatial_boxes)

                all_eigen_vectors_list.append(eigen_vectors)
                if test_images:
                    pass
                    # delayed(save_eigen_images)(eigen_vectors=eigen_vectors,
                    #                            output_dir=test_output_dir,
                    #                            image_shape=spatial_box_data.shape,
                    #                            box_num=spatial_box.box_num).compute()

        else:
            for temporal_box_num in range(len(image_data_filtered)):
                all_eigen_vectors_list.append(
                    loadEigenVectors(spatial_box_num=spatial_box.box_num,
                                     time_box_num=temporal_box_num,
                                     save_dir=save_dir))

        all_eigen_vectors = delayed(np.hstack)(all_eigen_vectors_list)
        all_boxes_eigen_vectors.append(all_eigen_vectors)
        if save_embedding_images:
            all_eigen_vectors = saveEmbedingNormImage(e_vectors=all_eigen_vectors,
                                                      image_shape=spatial_box.shape,
                                                      save_dir=save_dir,
                                                      spatial_box_num=spatial_box.box_num)

        data_all = spatial_box.extract_box(stack_crop_slice_data(image_data,slicing, crop))

        rois = roi_extract_image(e_vectors=all_eigen_vectors,
                                 original_shape=data_all.shape,
                                 original_2d_vol=reshape_to_2d_over_time(
                                     data_all),
                                 merge=merge,
                                 num_rois=num_rois, refinement=refinement,
                                 num_eigen_vector_select=num_eigen_vector_select,
                                 max_iter=max_iter,
                                 roi_size_min=roi_size_min,
                                 fill_holes=fill_holes,
                                 elbow_threshold_method=elbow_threshold_method,
                                 elbow_threshold_value=elbow_threshold_value,
                                 eigen_threshold_method=eigen_threshold_method,
                                 eigen_threshold_value=eigen_threshold_value,
                                 merge_temporal_coef=merge_temporal_coef,
                                 roi_size_limit=roi_size_max,
                                 box_num=spatial_box.box_num, print_progress=True,
                                 total_num_time_steps=total_num_time_steps,
                                 total_num_spatial_boxes=total_num_spatial_boxes,
                                 roi_eccentricity_limit=roi_eccentricity_limit,
                                 save_dir=save_dir, local_max_method=local_max_method,
                                 progress_signal=progress_signal)
        if test_images:
            pass
            # delayed(save_roi_images)(
            #     roi_list.json=spatial_box.redefine_spatial_cord_1d(rois),
            #                              output_dir=test_output_dir,
            #                              image_shape=shape,
            #                              box_num=spatial_box.box_num).compute()
        all_rois.append(spatial_box.redefine_spatial_cord_1d(rois))
    all_rois = delayed(reduce)(lambda x, y: x + y, all_rois)
    all_rois = all_rois.compute()
    data_all = stack_crop_slice_data(image_data, slicing, crop)

    all_rois_merged = delayed(merge_rois)(roi_list=all_rois,
                                          temporal_coefficient=merge_temporal_coef,
                                          original_2d_vol=reshape_to_2d_over_time(
                                              data_all)).compute()

    if test_images:
        delayed(save_roi_images)(roi_list=all_rois_merged,
                                 output_dir=test_output_dir,
                                 image_shape=shape, box_num="all").compute()
    if save_embedding_images and save_intermediate_steps:
        createEmbedingNormImageFromMultiple(spatial_box_list=spatial_boxes,
                                            save_dir=save_dir,
                                            num_time_steps=len(image_data_filtered))

    return all_rois_merged


if __name__ == '__main__':
    process_data(num_threads=1, load_data=True,
                 data_path="/Users/sschickler/Code Devel/LSSC-python/input_images" +
                           "/small_dataset1.tif",
                 test_images=True,
                 save_dir="/Users/sschickler/Code Devel/LSSC-python/output_images/15",
                 save_intermediate_steps=False,
                 eigen_vectors_already_generated=False,
                 save_embedding_images=False,
                 test_output_dir="/Users/sschickler/Code Devel/LSSC-python/output_images/15",
                 image_data_filtered=None,
                 total_num_time_steps=4, total_num_spatial_boxes=4, spatial_overlap=10,
                 filter=True, median_filter_size=(1, 3, 3), median_filter=True,
                 z_score=False, slice_stack=False, slice_every=10, slice_start=0,
                 metric="l2", knn=50, accuracy=59, connections=60,
                 num_eig=50, normalize_w_k=2, merge=True,
                 num_rois=25, refinement=True,
                 num_eigen_vector_select=5,
                 max_iter=400, roi_size_min=30,
                 fill_holes=True,
                 elbow_threshold_method=True,
                 elbow_threshold_value=1,
                 eigen_threshold_method=True,
                 eigen_threshold_value=.5,
                 merge_temporal_coef=.01,
                 roi_size_max=600)
    # with performance_report(filename="dask-report.html"):
    # process_data(num_threads=1, load_data=True,
    #              data_path="/Users/sschickler/Code Devel/LSSC-python/input_images/dataset_1",
    #              test_images=False,
    #              test_output_dir="/Users/sschickler/Documents/LSSC-python/output_images/16",
    #              image_data=None,
    #              save_dir="/Users/sschickler/Code DEvel/LSSC-python/output_images/16",
    #              save_embedding_images=True,
    #              save_intermediate_steps=True,
    #              eigen_vectors_already_generated=False,
    #              total_num_time_steps=20, total_num_spatial_boxes=16, spatial_overlap=30,
    #              filter=True, median_filter_size=(1, 3, 3), median_filter=True,
    #              z_score=False, slice_stack=False, slice_every=10, slice_start=0,
    #              metric="l2", knn=50, accuracy=59, connections=60,
    #              num_eig=50, normalize_w_k=2, merge=True,
    #              num_rois=25, refinement=True,
    #              num_eigen_vector_select=5,
    #              max_iter=400, roi_size_min=30,
    #              fill_holes=True,
    #              elbow_threshold_method=True,
    #              elbow_threshold_value=1,
    #              eigen_threshold_method=True,
    #              eigen_threshold_value=.5,
    #              merge_temporal_coef=.01,
    #              roi_size_max=600)


@delayed
def sliceData(data, start_end):
    return data[0][start_end[0]:start_end[1], :, :]


@delayed
def convert_to_float(data):
    return data.astype(np.float32)
@delayed
def stack_crop_slice_data(image_data, slicing, crop):
    if crop:
        data_all = np.vstack(
            [x[slicing[0]::slicing[1], crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]] for
             x in image_data])
    else:
        data_all = np.vstack([x[slicing[0]::slicing[1], :, :] for x in image_data])
    return data_all