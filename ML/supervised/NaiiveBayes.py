import numpy as np


class GaussianNB:

    def __init__(self):
        """
        _classes: ndarray, class labels (obtained from y)
        _mean, _var: ndarray, ndarray; mean and variance of the Gaussian distribution.
            Shape is (n_classes, n_features). Initialized to 0s.
        _priors: ndarray, prior probabilities. Shape is (n_classes,). Initial values 0s.
        """
        self._classes = None
        self._mean = None
        self._var = None
        self._priors = None

    def fit(self, X, y):
        """
        How does the fit work?

        :param X: ndarray, feature array of shape (n_samples, n_features)
        :param y: ndarray, target array of shape (n_samples, )
        :return: an instance of self
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # init mean, var, priors
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def gaussian_dist(self, class_idx, x):
        """
        Obtain a Gaussian distribution function for a given ???
        :param class_idx:
        :param x:
        :return: probability distribution function
        """
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        gauss = np.exp(-(x - mean) ** 2 / (2 * var))
        normal = 1. / np.sqrt(2 * np.pi * var)
        return normal * gauss

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self.gaussian_dist(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def predict(self, X):
        return [self._predict(x) for x in X]


# Naive Bayes Test
if __name__ == "__main__":

    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt


    def accuracy(y_true, y_model):
        return np.sum(y_true == y_model) / len(y_true)


    X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    print(X_test, '---', y_test.shape)

    nb = GaussianNB()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    print(f"Naive Bayes classification accuracy {100*accuracy(y_test, predictions):.2f}%")

    cnf_matrix = confusion_matrix(y_test, predictions)

    sns.heatmap(cnf_matrix, annot=True)
    plt.show()
