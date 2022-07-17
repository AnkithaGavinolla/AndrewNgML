import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from numpy.linalg import inv


class LinearRegression:

    def predict(self, features, weights):
        return np.dot(features, weights)

    def compute_cost(self, features, labels, weights):
        predictions = self.predict(features, weights)
        M = labels.shape[0]
        return np.sum(np.square(predictions - labels)) / (2 * M)

    def update_weights(self, weights, features, labels, lr):
        predictions = self.predict(features, weights)
        gradient = np.matmul(features.transpose(), predictions - labels)
        M = labels.shape[0]
        weights = weights - (lr / M) * gradient
        return weights

    def train(self, features, labels, weights, lr, iterations):
        costs_history = []
        for i in range(iterations):
            weights = self.update_weights(weights, features, labels, lr)
            cost = self.compute_cost(features, labels, weights)
            costs_history.append((i, cost))
        return weights, costs_history

    def train_normal_equation(self, features, labels):
        return np.dot(inv(np.matmul(features.transpose(), features)), np.matmul(features.transpose(), labels))



# Data1 = np.loadtxt('ex1/ex1data1.txt', delimiter=',')
# print("Plotting Data")
# plt.scatter(Data1[:, 0], Data1[:, 1], c='r', marker='x')
# plt.title("Data Plot")
# plt.show()
# labels = Data1[:, 1].reshape(-1, 1)
# features = np.append(np.ones((labels.shape[0], 1)), Data1[:, -1].reshape(-1, 1), axis=1)
# weights = np.zeros((features.shape[1], 1))
linear_regression = LinearRegression()
#
# weights, cost_history = linear_regression.train(features, labels, weights, 0.01, 1500)
# print("Printing weights = ", weights)
# print("Computing cost = ", cost_history)
# X = np.array([[1, 0], [1, 5], [1, 7.5], [1, 10.0], [1, 12.5], [1, 15.0], [1, 17.5], [1, 20.0], [1, 22.5], [1, 25.0]])
# plt.plot([X[i][1] for i in range(len(X))], [linear_regression.predict(X[i], weights) for i in range(len(X))])
# plt.plot([cost_history[i][0] for i in range(len(cost_history))], [cost_history[i][1] for i in range(len(cost_history))])
# plt.title("Training cost with lr = 0.01")
# plt.show()
#
#
# weights, cost_history = linear_regression.train(features, labels, weights, 0.1, 10)
# print("Printing weights = ", weights)
# print("Computing cost = ", cost_history)
# X = np.array([[1, 0], [1, 5], [1, 7.5], [1, 10.0], [1, 12.5], [1, 15.0], [1, 17.5], [1, 20.0], [1, 22.5], [1, 25.0]])
# plt.plot([X[i][1] for i in range(len(X))], [linear_regression.predict(X[i], weights) for i in range(len(X))])
# plt.plot([cost_history[i][0] for i in range(len(cost_history))], [cost_history[i][1] for i in range(len(cost_history))])
# plt.title("Training cost with lr = 0.1")
# plt.show()

Data2 = np.loadtxt('ex1/ex1data2.txt', delimiter=',')
X = Data2[:, :-1]
labels =Data2[:, -1].reshape(-1,1)
print(labels.shape)
standardized_X = preprocessing.scale(X)
features = np.append(np.ones((labels.shape[0], 1)), standardized_X, axis=1)
print(features.shape)
weights = np.zeros((features.shape[1], 1))
print(weights.shape)
weights, cost_history = linear_regression.train(features, labels, weights, 0.01, 1)