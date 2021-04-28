import numpy as np


class PCA:
    """
    Principal Component Analysis (PCA) allows us to represent high-dimensional data,
    using the dependencies between the features, in a more tractable low-dimensional
    form. This is one of the simplest, and most robust algorithm for dimensionality
    reduction.

    We want to project a p-dim vector space onto a q-dim (q<p) subspace. The basis
    vectors of this subspace are called principal component. The rule to project
    is to maximize variance (along the principal components). Equivalent algorithm
    is to look for projection with the smallest mean-squared distance between the
    original vectors and their projections on to the PCs.
    
    :parameter
        n_components: int; the number of dimensions to which the data is to be reduced
        X: list; the features and examples
        axis: 1 or 0 (default); sample x features (axis=0), features x samples (axis=1)
        verbose: bool, set True (default) to print all the PCs and some info. 
    """

    def __init__(self, n_components, X, axis=0, verbose=True):
        self.n_components = n_components
        self.X = np.asarray(X)
        self.axis = axis
        # axis=1: rows = features/variables, cols = examples/data
        # axis=0: rows = examples/data, cols = features/variables
        self.verbose = verbose

        self.components = None
        self.mean = None

    def principal_components(self):
        """
        Principal components of a dataset are the orthonormal directions along which
        the data is distributed. Eg., the major and minor axes of an ellipsoid-like
        distribution are its PCs. These are obtained from the eigen-vectors of the
        covariance matrix. The eigen-values on the other hand measure the ratio of
        the data present along the corresponding direction

        :return: self object; keeps the first n_components PCs.
        """
        self.mean = np.mean(self.X, axis=self.axis)

        # center the data
        self.X = self.X - self.mean

        # covariance matrix: np.cov has default axis=0
        cov_mat = np.cov(self.X) if self.axis else np.cov(self.X.T)

        eigenvalues, eigenvectors = np.linalg.eig(cov_mat)

        # ensure they are sorted in the decreasing order of e_values
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]

        # note that the col v[:,i] is the eigenvector corresponding to the
        # eigenvalue w[i] (numpy convention). Therefore, we transpose it
        # before applying the sorted args obtained above
        eigenvectors = eigenvectors.T
        eigenvectors = eigenvectors[idx]

        # keep the first n eigenvectors (components)
        self.components = eigenvectors[0: self.n_components]

        # R^2 of the projection = fitted variance / original variance
        r_squared = np.sum(eigenvalues[:self.n_components]) / np.sum(eigenvalues)
        r_squared = round(100 * r_squared, 3)

        if self.verbose:
            print("Data centroids along features: \n", tuple(np.round(self.mean, 2)))
            print("Covariance Matrix\n", np.round(cov_mat, 2))
            print("------ Principal components -------")
            print("PC_i | e_val | pct_data | e_vectors")
            print("...................................")
            for idx in range(len(eigenvalues)):
                print(f"PC_{idx} | ", np.round(eigenvalues[idx], 2),
                      "|", round(100 * eigenvalues[idx] / np.sum(eigenvalues), 2),
                      "% |", tuple(np.round(eigenvectors[idx], 2))
                      )
            print(f"\nR^2 is {r_squared}% for the first {self.n_components} PCs!")

    def projection(self):
        """
        projects data along the first n principal components
        
        :return: ndarray, projected data (onto the first n PCs) 
        """
        
        return np.dot(self.X, self.components.T)


# Driver code
if __name__ == '__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt
    
    # local needs, can be turned off
    import matplotlib
    matplotlib.use('Qt5Agg')

    data = datasets.load_iris()
    # has rows = data, cols = features (hence, axis=1)
    X = data.data
    y = data.target

    pca = PCA(n_components=2, X=X, axis=0)
    pca.principal_components()
    projected_data = pca.projection()

    x1 = projected_data[:, 0]
    x2 = projected_data[:, 1]
    plt.scatter(x1, x2, c=y, edgecolor='none', alpha=0.8)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()

    # # A simpler dataset (for debugging)
    # X = [[1, 2], [3, 3], [3, 5], [5, 4], [5, 6], [6, 5], [8, 7], [9, 8]]
    # # rows = examples (=8 data points) and cols = features (x1, x2)
    # pca = PCA(n_components=1, X=X, axis=0)
    # pca.principal_components()
