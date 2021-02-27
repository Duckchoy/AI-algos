import numpy as np


class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Finds a linear fit of the form y = w X + b

        Starting with the initial values (=zeros) of the parameters w (weight)
        and b (bias), it optimizes them using gradient descent over n_iters times.

        :param X: ndarray, feature array of shape (n_samples, n_features)
        :param y: ndarray, target array of shape (n_samples, )
        :return: an instance of self
        """
        n_samples, n_features = X.shape

        # init parameters (to zeros)
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            delta_w = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            delta_b = (1 / n_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.lr * delta_w
            self.bias -= self.lr * delta_b

    def predict(self, X):
        """
        Predicts the target values, y = w X + b, with the optimized parameters in self.
        :param X: ndarray, feature values of shape (var_size, n_features)
        :return: ndarray, target values of shape (var_size, )
        """
        return np.dot(X, self.weights) + self.bias


# Driver code
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    X, y = datasets.make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regress = LinearRegression(learning_rate=0.01, n_iters=1000)
    regress.fit(X_train, y_train)
    predictions = regress.predict(X_test)

    mse = np.mean((y_test - predictions) ** 2)
    print("MSE:", mse)

    y_pred_line = regress.predict(X)

    # Plotting the fit (works for n_features=1 only)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
    plt.show()
