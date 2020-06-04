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
        self.errors = []
        self.accuracies = []

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

        # online training
        for epoch in range(self.epochs):
            previous_delta = 0
            X, y = sklearn.utils.shuffle(X, y)
            y_outs = []
            for i in range(len(X)):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.sigmas)])
                y_out = a.T.dot(self.w) + self.b

                # loss = (y[i] - y_out) ** 2
                # print('Loss: {0:.2f}'.format(loss[0]))

                y_outs.append(y_out)

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

            self.errors.append(mean_squared_error(y, y_outs))
            y_rounded_outs = [np.round(num) for num in y_outs]
            n_differents = 0
            for i in range(len(y)):
                if y[i] != y_rounded_outs[i]:
                    n_differents += 1
            self.accuracies.append((len(y) - n_differents) / len(y))

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.sigmas)])
            y_out = a.T.dot(self.w) + self.b
            y_pred.append(y_out)
        return np.array(y_pred)

    def getErrors(self):
        return self.errors

    def getAccuracies(self):
        return self.accuracies


print("[1] Approximation\n[2] Classification")

choice = input("Please choose one of the options from above")

useKmeans = False
if choice == "1":
    trainSetChoice = input("Please choose train set between 1 and 2...")
    if trainSetChoice == "1":
        X1, y1 = read_data("approximation_train_1.txt")
    elif trainSetChoice == "2":
        X1, y1 = read_data("approximation_train_2.txt")
    else:
        print("Choice not available")
        sys.exit(0)
    X, y = read_data("approximation_test.txt")
elif choice == "2":
    X1, y1 = read_data("classification_train.txt")
    X, y = read_data("classification_test.txt")
    useKmeans = True
else:
    print("Choice not available")
    sys.exit(0)

number_of_neurons = [2, 8, 20]
momentum_factors = [0.1, 0.2, 0.5, 0.8]
learning_rates = [0.01, 0.05, 0.1, 0.3]

# number of neurons/centers k must be smaller than test data size

if choice == "1":
    for neorons in number_of_neurons:
        rbfnet = RBFNet(lr=1e-2, k=neorons, ifKmeans=useKmeans, withBias=True, beta=0.2)
        rbfnet.fit(X1, y1)
        y_pred = rbfnet.predict(X)
        plt.plot(X, y, '-,', label='true data')
        plt.plot(X, y_pred, '-,', label='RBFNN prediction')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        name = "aproximation_" + str(neorons) + "_neurons"
        plt.savefig("data/" + name + ".png")
        plt.clf()
        plt.plot(range(100), rbfnet.getErrors(), '-,')
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.tight_layout()
        name = "aproximation_" + str(neorons) + "_neurons_MSE"
        plt.savefig("data/" + name + ".png")
        plt.clf()
    for momentum in momentum_factors:
        rbfnet = RBFNet(lr=1e-2, k=10, ifKmeans=useKmeans, withBias=True, beta=momentum)
        rbfnet.fit(X1, y1)
        y_pred = rbfnet.predict(X)
        plt.plot(X, y, '-,', label='true data')
        plt.plot(X, y_pred, '-,', label='RBFNN prediction')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        name = "aproximation_" + str(momentum) + "_momentum"
        plt.savefig("data/" + name + ".png")
        plt.clf()
        plt.plot(range(100), rbfnet.getErrors(), '-,')
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.tight_layout()
        name = "aproximation_" + str(momentum) + "_momentum_MSE"
        plt.savefig("data/" + name + ".png")
        plt.clf()
    for lr in learning_rates:
        rbfnet = RBFNet(lr=lr, k=10, ifKmeans=useKmeans, withBias=True, beta=0.2)
        rbfnet.fit(X1, y1)
        y_pred = rbfnet.predict(X)
        plt.plot(X, y, '-,', label='true data')
        plt.plot(X, y_pred, '-,', label='RBFNN prediction')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        name = "aproximation_" + str(lr) + "_lr"
        plt.savefig("data/" + name + ".png")
        plt.clf()
        plt.plot(range(100), rbfnet.getErrors(), '-,')
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.tight_layout()
        name = "aproximation_" + str(lr) + "_lr_MSE"
        plt.savefig("data/" + name + ".png")
        plt.clf()
else:
    for neorons in number_of_neurons:
        rbfnet = RBFNet(lr=1e-2, k=neorons, ifKmeans=useKmeans, withBias=True, beta=0.2)
        rbfnet.fit(X1, y1)
        plt.plot(range(100), rbfnet.getAccuracies(), '-,')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.tight_layout()
        name = "classification_" + str(neorons) + "_neurons_accuracy"
        plt.savefig("data/" + name + ".png")
        plt.clf()
        plt.plot(range(100), rbfnet.getErrors(), '-,')
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.tight_layout()
        name = "classification_" + str(neorons) + "_neurons_MSE"
        plt.savefig("data/" + name + ".png")
        plt.clf()
    for momentum in momentum_factors:
        rbfnet = RBFNet(lr=1e-2, k=10, ifKmeans=useKmeans, withBias=True, beta=momentum)
        rbfnet.fit(X1, y1)
        plt.plot(range(100), rbfnet.getAccuracies(), '-,')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.tight_layout()
        name = "classification_" + str(momentum) + "_momentum_accuracy"
        plt.savefig("data/" + name + ".png")
        plt.clf()
        plt.plot(range(100), rbfnet.getErrors(), '-,')
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.tight_layout()
        name = "classification_" + str(momentum) + "_momentum_MSE"
        plt.savefig("data/" + name + ".png")
        plt.clf()
    for lr in learning_rates:
        rbfnet = RBFNet(lr=lr, k=10, ifKmeans=useKmeans, withBias=True, beta=0.2)
        rbfnet.fit(X1, y1)
        plt.plot(range(100), rbfnet.getAccuracies(), '-,')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.tight_layout()
        name = "classification_" + str(lr) + "_lr_accuracy"
        plt.savefig("data/" + name + ".png")
        plt.clf()
        plt.plot(range(100), rbfnet.getErrors(), '-,')
        plt.xlabel('epochs')
        plt.ylabel('MSE')
        plt.tight_layout()
        name = "classification_" + str(lr) + "_lr_MSE"
        plt.savefig("data/" + name + ".png")
        plt.clf()



