import numpy as np


class KMeans:
    """K-means clustering algorithm

    Parameters
    ----------
    n_clusters : int
        The number of clusters (=number of centroids) to be formed.

    init : {'random' or 'init'}, default='random'
        'random' : centroids are chosen at random (without replacement) from the data
        'init' : the first k data points are assigned to be the centroids.

    seed_ : int, default=42
        The seed of the random number generator. Relevant for init='random'.

    verbose : {0, 1}, default=1
        Verbosity mode. 1 prints all information, 0 does not.

    tol : float, default=0.0001
        Relative tolerance with regards to L2-norm of the difference in cluster centers
        of two consecutive iteration to declare convergence.

    max_iter : int, default=300
        Maximum number of iterations for a single run of the k-means routine.

    Notes
    -----
    The average complexity of k-means is O(k m T), where m is the number of examples
    and T is the number of iterations.

    The steps involved are the following:
        1. Assign (randomly/first k) data points as centroids (centroid initialization)
        2. Find distance of a point from all centroids; assign it to the
            least L2-distant centroid (cluster initialization). Repeat for all examples.
        3. Find the arithmetic mean of each cluster and make them the new centroids.
        4. Return to step 2, and repeat for max_iter number of iterations.
    """

    def __init__(self, n_clusters, init='random', seed_=42, verbose=1, tol=1e-4, max_iter=300):
        self.k = n_clusters
        self.init = init
        self.seed_ = seed_
        self.verbose = verbose
        self.tol = tol
        self.max_iter = max_iter

        self.centroids = np.array([])
        self.classifications = np.array([])
        self.n_features = 0
        self.n_examples = 0

    def init_centroids(self, X):
        # Initialize the centroids. For the zeroth iteration, they are medoids.

        self.n_examples, self.n_features = X.shape

        # The number of clusters should not exceed the number of data points.
        if self.n_examples < self.k:
            raise ValueError(f"Number of samples ({self.n_examples}) should be "
                             f">= number of clusters ({self.k}).")

        if self.init == 'init':
            # Pick the first 'k' data points as medoids.
            self.centroids = X[:self.k]
        else:
            import random
            # Pick 'k' points at random, without replacement.
            generator_ = np.random.default_rng(seed=self.seed_)
            self.centroids = generator_.choice(X, size=self.k, replace=False)

        if self.verbose:
            print("The initial choice of centroids are:")
            for k in range(self.k):
                print(f"Centroid {1 + k}: {self.centroids[k]}")

    @staticmethod
    def _dist_from_centroids(centroids, X):
        """
        A helper function that computes the distance of a point in X, from a centroid
        in the list 'centroids'.

        Parameters
        ----------
        centroids : ndarray of shape (n_clusters, n_features)
        X : ndarray of shape (n_examples, n_features)

        Returns
        -------
        dist_matrix : ndarray of shape (n_examples, n_clusters)
        """
        dist_matrix = []
        # Making use of Numpy's broadcasting, the distances of each data
        # from the centroids are obtained in a matrix: n_examples x n_clusters

        for centroid in centroids:
            dist_matrix.append(np.linalg.norm(X - centroid, axis=1, ord=2))

        # The numpy L2-distance matrix
        dist_matrix = np.asarray(dist_matrix).T

        # asserting rows = number of examples; cols = number of centroids
        assert dist_matrix.shape[0] == X.shape[0]
        assert dist_matrix.shape[1] == centroids.shape[0]

        return dist_matrix

    def _assign_clusters(self, X):
        # The centroid with the min L2 distance from a data is assigned to it.

        dist_matrix = self._dist_from_centroids(self.centroids, X)
        self.classifications = np.argmin(dist_matrix, axis=1)

    @staticmethod
    def get_centroids(cluster_labels, X):
        """
        A helper function to compute a new set of centroids for a given cluster label.
        The new centroids are simply the arithmetic mean of all the data points with the
        same cluster label. This happens to (theorem) minimize the sum of L2-distance
        of these data points from the new centroid (mean).

        Parameters
        ----------
        cluster_labels : ndarray of shape (n_examples, )
        X : ndarray of shape (n_examples, n_features)

        Returns
        -------
        new_centroids : ndarray of shape (n_clusters, n_features)

        """

        assert len(cluster_labels) == X.shape[0]
        assert cluster_labels.dtype == 'int64'

        # Find the unique cluster labels (throw-away) and the number of data points in them
        _, counts = np.unique(cluster_labels, return_counts=True)

        # Sum the values of each feature vector corresponding to each cluster.
        new_centroids = []
        for feature in range(X.shape[1]):
            # This is similar to what Panda's 'pd.groupby.mean' does, but we keep it "pure".
            new_centroids += [np.bincount(cluster_labels, weights=X[:, feature])]
            # Note, np.bincount works only for integer labels (hence, the assertion above)

        # Obtain the mean by dividing the summations with the population of each cluster.
        new_centroids = np.asarray(new_centroids / counts)

        return new_centroids.T

    def fit(self, X):
        """ Compute k-means clustering.

        Returns
        -------
        self
            Check class.centroids and class.classification for the fitted
            centroid coordinates and the cluster label of each example.
        """

        # Initialize the centroids
        self.init_centroids(X)

        for itr in range(1, self.max_iter):
            prev_centroids = self.centroids
            self._assign_clusters(X)

            if self.verbose:
                print("\niteration: ", itr, '\n--------------')
                print("centers:\n", self.centroids)
                print("cluster labels:\n", self.classifications)

            self.centroids = self.get_centroids(self.classifications, X)
            curr_centroids = self.centroids

            # If the new centroid does not move much from the previous one then
            # declare convergence (dictated by self.tol).
            optimized = np.sum(np.linalg.norm(curr_centroids - prev_centroids,
                                              axis=1, ord=2)) < self.tol
            if optimized:
                print("\nConvergence is achieved at iteration", itr)
                break

        if not optimized:
            print("\n[WARNING] Reached maximum iterations before reaching convergence, "
                  "consider increasing 'max_iter'.")

        # Finally, update the cluster labels for the optimized centroids.
        self._assign_clusters(X)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    # Dataset for testing
    X, y = make_blobs(n_samples=1500, random_state=170)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()

    # # Dataset for debugging
    # X = np.array([[1, 2], [1.5, 1.8], [5, 8 ], [8, 8], [1, 0.6], [9, 11]])
    # plt.scatter(X[:,0], X[:,1])
    # plt.show()

    model = KMeans(n_clusters=3, init='random', seed_=45, max_iter=100)

    model.fit(X)

    colors = 10 * ["g", "r", "c", "b", "k"]

    for m in range(model.n_examples):
        label_ = model.classifications[m]
        color = colors[label_]
        plt.scatter(X[m][0], X[m][1], marker="x", color=color, s=150, linewidths=5)

    for c in model.centroids:
        plt.scatter(x=c[0], y=c[1], marker="o", color="k", s=150)

    plt.show()