class Parameters:

    def __init__(self, metric="l2", knn=20, accuracy=200, connections=40,
                 num_threads=10, num_eig=25):
        """
        Initializes a parameter object
        Parameters
        ----------
        metric can be "l2" squared l2, "ip" Inner product, "cosine" Cosine similarity
        knn number of nearest neighbors to search for
        accuracy time of construction vs accuracy trade off
        connections max number of outgoing connections
        num_threads number of threads to run on
        num_eig number of eigen vectors to calculate
        """
        self.metric = metric
        self.knn = knn
        self.accuracy = accuracy
        self.connections = connections
        self.num_threads = num_threads
        self.num_eig = num_eig
