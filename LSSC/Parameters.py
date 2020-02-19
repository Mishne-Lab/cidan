from typing import Union, Any, List, Optional, cast, Tuple, Dict


class Parameters:

    def __init__(self, num_threads: int = 10,
                 bounding_box: bool = False,
                 bounding_box_val: Tuple[Tuple[int]] = ((0, 0, 0), (0, 0, 0)),
                 median_filter: bool = False,
                 median_filter_size: Tuple[int] = (1, 3, 3),
                 z_score: bool = False, slice_stack: bool = False,
                 slice_every: int = 1, slice_start: int = 0,
                 metric: str = "l2", knn: int = 50, accuracy: int = 200,
                 connections: int = 50, K: int = 2,
                 num_eig: int = 25, refinement: bool = True,
                 cluster_size_threshold: int = 30,
                 cluster_size_limit: int = 500,
                 fill_holes: bool = True,
                 num_eigen_vector_select: int = 5,
                 eigen_threshold_method: int = True,
                 eigen_threshold_value: float = .5,
                 elbow_threshold_method: bool = True,
                 elbow_threshold_multiplier: float = .9,
                 max_iter: int = 1000,
                 num_clusters: int = 100,
                 merge_temporal_coef: float = .1
                 ):
        """
        Parameters
        ----------
        num_threads: number of threads to run on
        bounding_box: Whether to use a bounding box to select a smaller area of image
         used by Stack_wrapper class
        bounding_box_val: A Tuple of 2 3D points for the bounding box (x,y, time)
        median_filter: Whether to apply median filter to original image
        median_filter_size: Size of median filter, should be 3D tuple
        z_score: Whether to z_score images
        slice_stack: used if the stack has multiple stacks embedded inside it
        slice_every: how often to select an image from the stack
        slice_start: starting image to slice
        metric: can be "l2" squared l2, "ip" Inner product, "cosine" Cosine similarity
        knn: number of nearest neighbors to search for
        accuracy time: of construction vs accuracy trade off
        connections: max number of outgoing connections
        K: used in autotune algorithm, decides which K closest neighbor to use
        num_eig: number of eigen vectors to calculate
        refinement: whether to preform refinement step in clustering algorithm
        cluster_size_threshold: Size threshold that all clusters need to be above
        cluster_size_limit: max size of cluster set to large number if don't want to exclude any
        fill_holes: whether to fill inner holes in each cluster
        num_eigen_vector_select: Number of eigen vectors to project into for
         clustering algorithm not used if threshold_method
        eigen_threshold_method: Way of determining eigen vectors to project
         into by only taking ones more than threshold*first eigenvector
        eigen_threshold_value: Value for eigen threshold
        elbow_threshold_method: new method for determining which pixels in cluster by only
         taking ones closer than elbow, only used in cluster refinement step
        elbow_threshold_multiplier: Multiplies the threshold determined by
         elbow method to either allow more pixels in when > 1 or
         or less when < 1
        max_iter: Max number of iterations to try in clustering algorithm
        num_clusters: number of clusters to try and generate, might not if runs
         over max_iter
        merge_temporal_coef: Coefficient for comparing if two clusters are
         similar can be 0-1
        """
        # General Parameters
        assert num_threads >= 1
        self.num_threads = num_threads

        # Filter function parameters
        assert isinstance(bounding_box, bool)
        assert (bounding_box and len(bounding_box_val) == 2 and all(
            [len(x) == 3 for x in bounding_box_val])) or not bounding_box
        self.bounding_box = bounding_box
        self.bounding_box_val = bounding_box_val
        assert isinstance(median_filter, bool)
        self.median_filter = median_filter
        assert isinstance(median_filter_size, tuple) and \
               all([isinstance(x, int) for x in median_filter_size]) and \
               len(median_filter_size) == 3
        self.median_filter_size = median_filter_size
        assert isinstance(z_score, bool)
        self.z_score = False
        assert isinstance(slice_stack, bool)
        self.slice = slice_stack
        assert isinstance(slice_every, int) and slice_every >= 1
        self.slice_every = slice_every
        assert isinstance(slice_start, int) and slice_start >= 0
        self.slice_start = slice_start
        if median_filter or z_score:
            self.filter = True
        else:
            self.filter = False

        # Affinity matrix parameters
        assert knn < accuracy, "Knn needs to be less than the accuracy amount"
        assert K <= knn, "K needs to be less than the knn amount"
        assert isinstance(metric, str) and metric in ["l2", "ip", "cosine"]
        self.metric = metric
        assert isinstance(knn, int) and knn >= 1
        self.knn = knn
        assert isinstance(accuracy, int) and accuracy >= 1
        self.accuracy = accuracy
        assert isinstance(connections, int) and connections >= 1
        self.connections = connections
        assert isinstance(K, int) and K >= 0
        self.K = K

        # Eigen Vector Generation Parameters1
        assert isinstance(num_eig, int) and num_eig >= 1
        self.num_eig = num_eig

        # Clustering Parameters
        assert isinstance(refinement, bool)
        self.refinement = refinement
        assert isinstance(cluster_size_threshold,
                          int) and cluster_size_threshold >= 0
        self.cluster_size_threshold = cluster_size_threshold
        assert isinstance(cluster_size_limit,
                          int) and cluster_size_limit >= cluster_size_threshold
        self.cluster_size_limit = cluster_size_limit
        assert isinstance(fill_holes, bool)
        self.fill_holes = fill_holes
        assert isinstance(num_eigen_vector_select,
                          int) and num_eigen_vector_select >= 0
        self.num_eigen_vector_select = num_eigen_vector_select
        assert isinstance(eigen_threshold_method, bool)
        self.eigen_threshold_method = eigen_threshold_method
        assert isinstance(eigen_threshold_value,
                          float) and 1 > eigen_threshold_value > 0
        self.eigen_threshold_value = eigen_threshold_value
        assert isinstance(elbow_threshold_method, bool)
        self.elbow_threshold_method = elbow_threshold_method
        assert isinstance(elbow_threshold_multiplier, float) \
               and elbow_threshold_multiplier >= 0
        self.elbow_threshold_multiplier = elbow_threshold_multiplier
        assert isinstance(num_clusters, int) and num_clusters > 0
        self.num_clusters = num_clusters
        assert isinstance(max_iter, int) and max_iter > num_clusters
        self.max_iter = max_iter
        assert isinstance(merge_temporal_coef, float)
        assert 0 < merge_temporal_coef < 1
        self.merge_temporal_coef = merge_temporal_coef
