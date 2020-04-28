from CIDAN.LSSC.functions.data_manipulation import load_filter_tif_stack, filter_stack, \
    reshape_to_2d_over_time
from CIDAN.LSSC.functions.roi_extraction import roi_extract_image, merge_rois
from CIDAN.LSSC.functions.embeddings import calc_affinity_matrix
from CIDAN.LSSC.functions.pickle_funcs import pickle_save, pickle_load, pickle_clear, \
    pickle_set_dir, pickle_exist
from CIDAN.LSSC.functions.temporal_correlation import *
from CIDAN.LSSC.functions.eigen import gen_eigen_vectors, save_eigen_vectors, \
    load_eigen_vectors, save_embeding_norm_image, create_embeding_norm_multiple
from dask import delayed
from functools import reduce

import numpy as np
from typing import Union, Any, List, Optional, cast, Tuple, Dict
from CIDAN.LSSC.SpatialBox import SpatialBox
from CIDAN.LSSC.functions.save_test_images import save_eigen_images, save_volume_images, \
    save_roi_images

from dask.distributed import performance_report
import logging
logger1 =logging.getLogger("CIDAN.LSSC.process_data")
def process_data(*, num_threads: int, test_images: bool, test_output_dir: str,
                 save_dir: str, save_intermediate_steps: bool,
                 load_data: bool, data_path: str,
                 image_data: np.ndarray,
                 eigen_vectors_already_generated: bool,
                 save_embedding_images: bool,
                 total_num_time_steps: int, total_num_spatial_boxes: int,
                 spatial_overlap: int, filter: bool, median_filter: bool,
                 median_filter_size: Tuple[int],
                 z_score: bool, slice_stack: bool,
                 slice_every, slice_start: int, metric: str, knn: int,
                 accuracy: int, connections: int, normalize_w_k: int, num_eig=25,
                 merge: bool,
                 num_rois: int, refinement: bool, num_eigen_vector_select: int,
                 max_iter: int, roi_size_min: int, fill_holes: bool,
                 elbow_threshold_method: bool, elbow_threshold_value: float,
                 eigen_threshold_method: bool,
                 eigen_threshold_value: float, merge_temporal_coef: float,
                 roi_size_max: int):
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
                 roi_size_max {38}""".format(num_threads,test_images , test_output_dir,
                 save_dir, save_intermediate_steps,
                 load_data, data_path,
                 image_data,
                 eigen_vectors_already_generated,
                 save_embedding_images,
                 total_num_time_steps, total_num_spatial_boxes,
                 spatial_overlap, filter, median_filter,
                 median_filter_size,
                 z_score, slice_stack,
                 slice_every, slice_start, metric, knn,
                 accuracy, connections, normalize_w_k, num_eig,
                 merge,
                 num_rois, refinement, num_eigen_vector_select,
                 max_iter, roi_size_min, fill_holes,
                 elbow_threshold_method, elbow_threshold_value,
                 eigen_threshold_method,
                 eigen_threshold_value, merge_temporal_coef,
                 roi_size_max))
    # TODO Make after eigen vector make function to save intermediate embeding norm
    #  for each spatial box
    # TODO Rewrite to take in a list of loaded datasets

    # TODO add assertions to make sure input splits work for dataset
    if load_data == True:
        image = load_filter_tif_stack(path=data_path, filter=filter,
                                      median_filter=median_filter,
                                      median_filter_size=median_filter_size,
                                      z_score=z_score, slice_stack=slice_stack,
                                      slice_start=slice_start,
                                      slice_every=slice_every)
    else:
        image = image_data
    if test_images:
        save_volume_images(volume=image, output_dir=test_output_dir)
    shape = image.shape
    logger1.debug("image shape {0}".format(shape))
    print("Creating {} spatial boxes".format(total_num_spatial_boxes))
    spatial_boxes = [SpatialBox(box_num=x, total_boxes=total_num_spatial_boxes,
                                spatial_overlap=spatial_overlap, image_shape=shape)
                     for x in range(total_num_spatial_boxes)]
    all_rois = []
    all_boxes_eigen_vectors = []
    for spatial_box in spatial_boxes:
        spatial_box_data = spatial_box.extract_box(image)
        time_boxes = [(x * (shape[0] // total_num_time_steps), (x + 1) * (shape[0] //
                                                                          total_num_time_steps))
                      for x in range(total_num_time_steps)]
        all_eigen_vectors_list = []
        if not eigen_vectors_already_generated:
            for temporal_box_num, start_end in enumerate(time_boxes):
                start, end = start_end

                time_box_data = spatial_box_data[start:end, :, :]
                time_box_data_2d = reshape_to_2d_over_time(time_box_data)
                logger1.debug("Time box {0}, start {1}, end {2}, time_box shape {3}, 2d shape {4}".format(temporal_box_num, start,end,time_box_data.shape,time_box_data_2d.shape))
                k = calc_affinity_matrix(pixel_list=time_box_data_2d, metric=metric,
                                         knn=knn, accuracy=accuracy,
                                         connections=connections,
                                         normalize_w_k=normalize_w_k,
                                         num_threads=num_threads,spatial_box_num=spatial_box.box_num, temporal_box_num=temporal_box_num)
                eigen_vectors = gen_eigen_vectors(K=k,
                                                  num_eig=num_eig //
                                                          total_num_time_steps,spatial_box_num=spatial_box.box_num, temporal_box_num=temporal_box_num)
                if save_intermediate_steps:
                    eigen_vectors = save_eigen_vectors(e_vectors=eigen_vectors,
                                                       spatial_box_num=spatial_box.box_num,
                                                       time_box_num=temporal_box_num, save_dir= save_dir)

                all_eigen_vectors_list.append(eigen_vectors)
                if test_images:
                    pass
                    # delayed(save_eigen_images)(eigen_vectors=eigen_vectors,
                    #                            output_dir=test_output_dir,
                    #                            image_shape=spatial_box_data.shape,
                    #                            box_num=spatial_box.box_num).compute()

        else:
            for temporal_box_num in range(total_num_time_steps):
                all_eigen_vectors_list.append(load_eigen_vectors(spatial_box_num=spatial_box.box_num,
                                                                 time_box_num=temporal_box_num,
                                                                 save_dir=save_dir))

        all_eigen_vectors = delayed(np.hstack)(all_eigen_vectors_list)
        all_boxes_eigen_vectors.append(all_eigen_vectors)
        if save_embedding_images:
            all_eigen_vectors = save_embeding_norm_image(e_vectors=all_eigen_vectors,
                                                     image_shape=spatial_box.shape,
                                                     save_dir=save_dir,
                                                     spatial_box_num=spatial_box.box_num)

        rois = roi_extract_image(e_vectors=all_eigen_vectors,
                                 original_shape=spatial_box_data.shape,
                                 original_2d_vol=reshape_to_2d_over_time(
                                     spatial_box_data),
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
                                 roi_size_limit=roi_size_max, box_num=spatial_box.box_num)
        if test_images:
            pass
            # delayed(save_roi_images)(
            #     roi_list=spatial_box.redefine_spatial_cord_1d(rois),
            #                              output_dir=test_output_dir,
            #                              image_shape=shape,
            #                              box_num=spatial_box.box_num).compute()
        all_rois.append(spatial_box.redefine_spatial_cord_1d(rois))
    all_rois = delayed(reduce)(lambda x, y: x + y, all_rois)
    all_rois = all_rois.compute()
    all_rois_merged = delayed(merge_rois)(roi_list=all_rois,
                                              temporal_coefficient=merge_temporal_coef,
                                              original_2d_vol=reshape_to_2d_over_time(
                                                      image)).compute()

    if test_images:
        delayed(save_roi_images)(roi_list=all_rois_merged,
                                     output_dir=test_output_dir,
                                     image_shape=shape, box_num="all").compute()
    if save_embedding_images and save_intermediate_steps:
        create_embeding_norm_multiple( spatial_box_list=spatial_boxes,save_dir=save_dir, num_time_steps=total_num_time_steps)


    return all_rois_merged


if __name__ == '__main__':
    process_data(num_threads=1, load_data=True,
                 data_path="/Users/sschickler/Code Devel/LSSC-python/input_images" +
                           "/small_dataset.tif",
                 test_images=True,
                 save_dir="/Users/sschickler/Code Devel/LSSC-python/output_images/15",
                 save_intermediate_steps=False,
                 eigen_vectors_already_generated=False,
                 save_embedding_images=False,
                 test_output_dir="/Users/sschickler/Code Devel/LSSC-python/output_images/15",
                 image_data=None,
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