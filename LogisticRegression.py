import numpy as np


class LogisticRegression:

    def __init__(self, learn_rate=0.001, max_iter=1000, activation='sigmoid', print_cost=False):
        self.learn_rate = learn_rate
        self.max_iter = max_iter
        self.print_cost = print_cost
        self.activation = activation
        self.weights = None
        self.bias = None

    def activate(self, x):
        if self.activation == 'sigmoid':
            return 1. / (1. + np.exp(-x))

    @staticmethod
    def cost_fun(y_orig, y_pred):
        # Cost function = -ve log of the max likelihood [probability, p(y|x)] estimator
        cost = y_orig * np.log(y_pred) + (1 - y_orig) * np.log(1 - y_pred)
        # The log may cause trouble if data is not properly normalized/standardized
        return - np.sum(cost) / len(y_orig)

    def fit(self, X, y):

        n_examples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Optimization loop
        for itr in range(self.max_iter):
            # forward propagation ...
            # Establish the linear model: z = w x + b
            linear_model = np.dot(X, self.weights) + self.bias

            # Use an activation function to predict the class
            y_hat = self.activate(linear_model)

            # Compute the cost function
            cost = self.cost_fun(y, y_hat)
            # Print the cost (if activated) every 100 iterations
            if self.print_cost and itr % 100 == 0:
                print("Cost after iteration %i: %f" % (itr, cost))

            # (implement ReLU etc later)
            if self.activation == 'sigmoid':

                # Back propagation ...
                # dw := d(cost)/dw; db := d(cost)/db
                dw = (1 / n_examples) * np.dot(X.T, (y_hat - y))
                db = (1 / n_examples) * np.sum(y_hat - y)

                # Parameter updates via gradient descent
                self.weights -= self.learn_rate * dw
                self.bias -= self.learn_rate * db

        # The optimized parameters (weights, bias) are returned via the 'self' object

    def predict_class(self, X):
        # Predicted class = Heaviside(y_hat - 0.5)
        linear_model = np.dot(X, self.weights) + self.bias
        y_hat = self.activate(linear_model)
        y_hat_cls = [1 if i > 0.5 else 0 for i in y_hat]
        return y_hat_cls


# Driver code
if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler

    # Test Logistic regression in breast_cancer
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # We 'standardize' the examples by subtracting the mean, dividing the std(X)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regressor = LogisticRegression(learn_rate=0.0001, max_iter=1000, print_cost=True)
    regressor.fit(X_train, y_train)

    def accuracy(y_orig, y_pred):
        # An alternative method is: round(100 - np.mean(np.abs(y_orig - y_pred) * 100.), 4
        return round(np.mean(y_orig == y_pred) * 100, 4)

    y_train_hat = regressor.predict_class(X_train)
    print("Train Accuracy: ", accuracy(y_train, y_train_hat))

    y_test_hat = regressor.predict_class(X_test)
    print("Test Accuracy: ", accuracy(y_test, y_test_hat))
