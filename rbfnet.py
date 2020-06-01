import sys

import numpy as np
import matplotlib.pyplot as plt
import random

import sklearn


def read_data(data_path):
    data_file = open(data_path, "r")
    data = data_file.readlines()
    data_file.close()
    lines = []
    for line in data:
        lines.append(line.split(" "))
    X = []
    y = []
    for i in range(len(lines)):
        X.append([])
        for j in range(len(lines[i]) - 1):
            X[i].append(float(lines[i][j]))
        y.append(float(lines[i][-1]))
    X, y = zip(*sorted(zip(X, y)))
    return X, y

# def derivative_gauss_sigma(x, c, s):
#     return get_distance(x, c) ** 2 * rbf(x, c, s) / s ** 3
#
# def derivative_gauss_center(x, c, s):
#     return get_distance(x, c) * rbf(x, c, s) / s ** 2

def mean_squared_error(y, y_pred):
    sum = 0
    for i in range(len(y)):
        sum += (y[i] - y_pred[i]) ** 2
    return sum / len(y)

def rbf(x, c, s):
    return np.exp(-get_distance(x, c)**2 / (2 * s**2))

def get_distance(x1, x2):
    sum = 0
    for i in range(len(x1)):
        sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)


def kmeans(X, k):
    centroids = random.sample(X, k)     # randomly select k data point

    converged = False   # Flag to terminate process after convergence

    while (not converged):

        cluster_list = [[] for i in range(len(centroids))]  # cluster for each centeroid

        for x in X:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))   # get distance to each centeroid
            cluster_list[int(np.argmin(distances_list))].append(x)  # add for minimum distance

        cluster_list = list((filter(None, cluster_list)))   # remove clusters which are empty

        prev_centroids = centroids.copy()   # save centroids to compare later

        centroids = []

        for j in range(len(cluster_list)):
            centroids.append(np.mean(cluster_list[j], axis=0))  # calculate the new clusters

        pattern = np.abs(np.sum(prev_centroids) - np.sum(centroids))    # get rate of convergence

        converged = (pattern == 0)  # check for convergence

    return np.array(centroids)



class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2, lr=0.01, epochs=100, rbf=rbf, ifKmeans=False, withBias=True, beta=0.0):
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.rbf = rbf
        self.ifKmeans = ifKmeans
        self.withBias = withBias
        self.beta = beta

        self.w = np.random.randn(k)
        if self.withBias:
            self.b = np.random.randn(1)
        else:
            self.b = 0

    def fit(self, X, y):
        if(self.ifKmeans):
            self.centers = kmeans(X, self.k)
        else:
            self.centers = random.sample(X, self.k)
        dMaxArray = []
        for center in self.centers:
            dMax = max(get_distance(center, anotherCenter) for anotherCenter in self.centers)
            dMaxArray.append(dMax)
        self.sigmas = []
        for dMax in dMaxArray:
            self.sigmas.append(dMax / np.sqrt(2*self.k))

        previous_delta = 0

        # online training
        for epoch in range(self.epochs):
            X, y = sklearn.utils.shuffle(X, y)
            for i in range(len(X)):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.sigmas)])
                y_out = a.T.dot(self.w) + self.b

                # loss = (y[i] - y_out) ** 2
                # print('Loss: {0:.2f}'.format(loss[0]))

                # backward pass
                error = -(y[i] - y_out)

                # online update
                delta = -self.lr * error
                self.w = self.w + delta * a + self.beta * previous_delta
                
                if self.withBias:
                    self.b = self.b - self.lr * error + self.beta * previous_delta

                previous_delta = delta

                # TODO: back propagation for hidden layer
                # error1 = [error * w for w in self.w]
                # error1_sigma = np.multiply([derivative_gauss_sigma(x, c, s) for x, c, s, in zip(a, self.centers, self.sigmas)], error1)
                # error1_center = np.multiply([derivative_gauss_center(x, c, s) for x, c, s, in zip(a, self.centers, self.sigmas)], error1)

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.sigmas)])
            y_out = a.T.dot(self.w) + self.b
            y_pred.append(y_out)
        return np.array(y_pred)


print("[1] Approximation\n[2] Classification")

choice = input("Please choose one of the options from above")

useKmeans = False
if choice == "1":
    X1, y1 = read_data("approximation_train_1.txt")
    # X1, y1 = read_data("approximation_train_2.txt")
    X, y = read_data("approximation_test.txt")
elif choice == "2":
    X1, y1 = read_data("classification_train.txt")
    X, y = read_data("classification_test.txt")
    useKmeans = True
else:
    print("Choice not available")
    sys.exit(0)

rbfnet = RBFNet(lr=1e-2, k=20, ifKmeans=useKmeans, withBias=True, beta=0.2)
rbfnet.fit(X1, y1)
y_pred = rbfnet.predict(X)

if choice == "1":
    plt.plot(X, y, '-o', label='true data')
    plt.plot(X, y_pred, '-o', label='RBFNN prediction')
    plt.legend()
    print("Mean squared error is " + str(mean_squared_error(y, y_pred)))
    plt.tight_layout()
    plt.show()
else:
    y_pred = [np.round(num) for num in y_pred]
    n_differents = 0
    for i in range(len(y)):
        if y[i] != y_pred[i]:
            n_differents += 1
    print("Accuracy is " + str((len(y) - n_differents)/len(y)) + "[%]")



