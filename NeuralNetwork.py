import numpy as np


class NeuralNetwork:
    """
    This is a simple 2 layer neural network. It has 1 input layer with nx number
    of features, and a hidden layer with nh number of hidden units (nodes). The
    output layer has ny units (each containing 2 classes).
    The neural network is trained in the following manner
        (1) Initialization: the parameters (weights, biases) by random variables
            & zeros. See the method init_params()
        (2) Forward propagation: obtains the linear model (y=wx+b) and passes the
            model through various layers via an activator. See forward_prop().
            Since we assume a binary classification problem, the out layer has
            'sigmoid' activation function hard coded to it.
        (3) Cross-entropy: is computed, which is the cost function obtained as a
            mean of the loss functions of all the examples in the dataset. Loss
            function is -ve log of the max likelihood estimator. See cost_fun()
        (4) Backward propagation: allows us to traverse back to the input layer
            from the output layer (via chain rules of derivative), eventually
            allowing us to obtain d(cost)/d(weight) and d(cost)/d(bias).
        (5) Gradient descent: makes use of the above derivatives and updates the
            parameters that are passed on to forward propagate again (via a loop).
    After the training is done (using the above steps), the test sets are predicted
    by using the optimized parameters and performing one forward propagation.
    See the method, predict_class().
    """

    def __init__(self,
                 learn_rate=0.001,
                 max_iter=1000,
                 hidden_layers=4,
                 seed=42,
                 feature_axis=0,
                 activation='tanh',
                 print_cost=False
                 ):

        # Hyper-parameters
        self.lr = learn_rate                # (float) learning rate
        self.max_iter = max_iter            # (int) max number of iterations in optimization
        self.nh = hidden_layers             # (int) number of hidden layers
        self.seed = seed                    # (int) fix the random state
        self.feature_axis = feature_axis    # (0, 1) the features are in a row (0) or a column (1)
        self.print_cost = print_cost        # (bool) prints the cost (cross-entropy) function
        self.activation = activation        # 'tanh', 'sigmoid', 'ReLU'

        # Optimization parameters
        self.W1 = None                      # weights in layer 1
        self.b1 = None                      # biases in layer 1
        self.W2 = None                      # weights in layer 2
        self.b2 = None                      # biases in layer 2

    # The activation function: returns the function and its gradient
    @staticmethod
    def activate(x, key):

        if key == 'sigmoid':
            a_func = 1 / (1 + np.exp(-x))
            a_grad = np.multiply(a_func, 1-a_func)

        elif key == 'tanh':
            a_func = np.tanh(x)
            a_grad = 1 - np.power(a_func, 2)

        elif key == 'ReLU':
            a_func = np.maximum(0, x)
            a_grad = np.heaviside(x, 0)  # x==0 returns 0

        else:
            raise KeyError("Invalid activation key. "
                           "Choose from 'tanh', 'sigmoid', 'ReLU'")

        return a_func, a_grad

    # Cost function = -ve log of the max likelihood [probability, p(y|x)] estimator
    @staticmethod
    def cost_fun(y_orig, y_hat):

        log_prob = np.multiply(np.log(y_hat), y_orig) + np.multiply(
            (1 - y_orig), np.log(1 - y_hat))

        cost = - np.mean(log_prob)
        cost = np.squeeze(cost)

        return cost

    def init_params(self, num_in_units, num_out_units):
        """
        Initialize the weights to random numbers (for "breaking the symmetry").
        Setting weights = 0 makes all the hidden units spit out the same params.
        A 'scale' is multiplied to push the initial values close to zero. This
        makes the gradient descent work converge faster.
        The biases are initialized to zero.

        :param num_in_units: int number of input units (features)
        :param num_out_units: int number of output units (classes)
        :return: 'self' object (updates the weights, and biases)
        """

        np.random.seed(self.seed)  # random variables are frozen

        scale = 0.01  # scale the weights to speed up gradient descent

        self.W1 = np.random.randn(self.nh, num_in_units) * scale
        self.b1 = np.zeros(shape=(self.nh, 1))
        self.W2 = np.random.randn(num_out_units, self.nh) * scale
        self.b2 = np.zeros(shape=(num_out_units, 1))
        # print(self.b1.shape, self.W1.shape, self.b2.shape, self.W2.shape)

    def forward_prop(self, X):

        Z1 = np.dot(self.W1, X) + self.b1
        A1, grad_A1 = self.activate(Z1, key=self.activation)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2, _ = self.activate(Z2, key='sigmoid')
        # grad_A2 is redundant since (for the moment) the activation
        # function for the output layer is hard coded to be sigmoid.

        return A1, grad_A1, A2

    def fit(self, X, Y):

        if self.feature_axis:
            X, Y = X.T, Y.T

        nx, n_examples = X.shape
        ny = 1 if len(Y.shape) == 1 else Y.shape[0]

        # ------ 1. INITIALIZATION ------ #
        self.init_params(nx, ny)

        costs = []  # The cost will be appended here for every iteration

        for itr in range(1 + self.max_iter):

            # ------ 2. FORWARD PROPAGATION ------ #
            A1, grad_A1, A2 = self.forward_prop(X)

            # ----- 3. COST FUNCTION ----- #
            if self.print_cost:
                costs.append(self.cost_fun(Y, A2))
                # Print the cost (if activated) every few iterations
                if itr % (self.max_iter // 10) == 0:
                    print("Cost after iteration %i: %f" % (itr, costs[itr]))

            # ----- 4. BACK PROPAGATION ----- #
            dZ2 = A2 - Y
            dW2 = np.dot(dZ2, A1.T) / n_examples
            db2 = np.sum(dZ2, axis=1, keepdims=True) / n_examples
            dZ1 = np.multiply(np.dot(self.W2.T, dZ2), grad_A1)
            dW1 = np.dot(dZ1, X.T) / n_examples
            db1 = np.sum(dZ1, axis=1, keepdims=True) / n_examples

            # ----- 5. GRADIENT DESCENT ------ #
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict_class(self, X):

        if self.feature_axis:
            X = X.T

        _, __, y_hat = self.forward_prop(X)

        return np.round(y_hat)


# Driver code: There are two datasets to work on. They are not linearly separable
# hence, logistic regression does not work. Note, use the 'feature_axis' key to
# ensure the features and the examples are not confused.
if __name__ == "__main__":

    # Check accuracy of the fit
    def accuracy(y_orig, y_pred):
        return round(np.mean(y_orig == y_pred) * 100, 4)

    # An in-house train-test splitter
    def train_test(A, B, test_size=0.2):

        nums = A.shape[1]
        frac = round(nums * (1 - test_size))

        # Shuffle the index array and then map that to X, Y
        idx = np.arange(nums)
        np.random.shuffle(idx)
        A = A[:, idx]
        B = B[:, idx]

        return A[:, :frac], A[:, frac:], B[:, :frac], B[:, frac:]

    # --------- DATASET I -------- #
    # planar_utils dataset is available here: https://bit.ly/3vfAqef
    from planar_utils import *
    import matplotlib.pyplot as plt

    nn = NeuralNetwork(
        learn_rate=0.5, max_iter=10000, hidden_layers=5, seed=2,
        activation='tanh', print_cost=True
        )

    X, y = load_planar_dataset()
    plt.scatter(X[0, :], X[1, :], c=y, s=40)
    plt.show()

    X_train, X_test, y_train, y_test = train_test(X, y, test_size=0.2)
    nn.fit(X_train, y_train)

    # Plot the decision boundary
    plot_decision_boundary(lambda x: nn.predict_class(x.T), X_train, y_train)
    plt.title("Decision Boundary")
    plt.show()

    y_train_hat = nn.predict_class(X_train)
    print("Train Accuracy: ", accuracy(y_train, y_train_hat))

    y_test_hat = nn.predict_class(X_test)
    print("Test Accuracy: ", accuracy(y_test, y_test_hat))

    # --------- DATASET II -------- #
    # from sklearn.datasets import make_circles
    # from sklearn.model_selection import train_test_split
    #
    # X, y = make_circles(factor=0.5, random_state=0, noise=0.05)
    #
    # plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolor='k')
    # plt.show()
    #
    # nn = NeuralNetwork(
    #     learn_rate=0.5, max_iter=1000, hidden_layers=5, seed=2,
    #     feature_axis=1, activation='tanh', print_cost=True
    #     )
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    #
    # nn.fit(X_train, y_train)
    #
    # y_train_hat = nn.predict_class(X_train)
    # print("Train Accuracy: ", accuracy(y_train, y_train_hat))
    #
    # y_test_hat = nn.predict_class(X_test)
    # print("Test Accuracy: ", accuracy(y_test, y_test_hat))
