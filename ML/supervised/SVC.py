# Support Vector Classifier (SVC)

import numpy as np


class SVC:
    """
    A robust (to outliers) large/maximum margin classifier: maximize the size of
    the projection of a data point onto the parameter axis (which sits normal to
    the decision boundary). In other words, minimize ||params||.
    """
    
    def __init__(self, learning_rate=0.001, lagrange=0.01, epochs=1000, print_cost=True):
        self.lr = learning_rate
        self.lagrange = lagrange
        self.epochs = epochs
        self.print_cost = print_cost
        self.w = None
        self.b = None

    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initialize the parameters
        self.w = np.zeros(n_features)
        self.b = 0

        # Heaviside function: return -1 (+1) for all -ve (+ve) y-values.
        class_score = np.where(y <= 0, -1, 1)

        for epoch in range(self.epochs):
            hinge_loss = 0
            # This is not a vectorized implementation
            for i, x_i in enumerate(X):
                # The distance of x_i to the decision hyperplane
                distance = class_score[i] * (np.dot(x_i, self.w) - self.b) - 1

                # canonical representation of the decision hyperplane
                condition = distance >= 0

                if condition:
                    hinge_loss -= 0
                    self.w -= self.lr * (2 * self.lagrange * self.w)
                else:
                    hinge_loss -= distance
                    self.w -= self.lr * (2 * self.lagrange * self.w - np.dot(x_i, class_score[i]))
                    self.b -= self.lr * class_score[i]

            # ----- Cost function ----- #
            if self.print_cost:
                cost = self.lagrange * np.linalg.norm(self.w)**2 + hinge_loss
                # Print the cost (if activated) every few iterations
                if epoch % (self.epochs // 10) == 0:
                    print("Cost after iteration %i: %f" % (epoch, cost))

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)


# Driver code
if __name__ == '__main__':
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_blobs(n_samples=50, n_features=3, centers=3,
                               cluster_std=1.05, random_state=40)
    y = np.where(y == 0, -1, 1)

    clf = SVC()
    clf.fit(X, y)
    predictions = clf.predict(X)

    print(clf.w, clf.b)

  # Visualization of the decision planes
    def visualize_svc():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], 'y--')
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], 'k')
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], 'k')

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()


    visualize_svc()
