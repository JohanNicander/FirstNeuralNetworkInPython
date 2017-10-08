import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import os


def johan():
    # Read data
    path = os.path.dirname(os.path.abspath(__file__))
    indata = open(path + r'/Data.txt')
    dataframe = pd.read_fwf(indata)
    x_values = dataframe[['X']]
    y_values = dataframe[['Y']]
    print(x_values)

    # Liear regression
    reg = linear_model.LinearRegression()
    reg.fit(x_values, y_values)

    # Visulize results
    plt.scatter(x_values, y_values)
    plt.plot(x_values, reg.predict(x_values))
    plt.show()


class NeuralNet:
    def __init__(self, neuralshape=np.array([1, 1])):
        self.setNeuralShape(neuralshape)
        self.setNeuralStructure(self.neuralshape)

    def setNeuralShape(self, neuralshape=np.array([1, 1])):
        if type(neuralshape) is np.ndarray and neuralshape.ndim == 1:
            self.neuralshape = neuralshape
        else:
            self.neuralshape = np.array([1, 1])
        self.setNeuralStructure(neuralshape)

    def getNeuralShape(self):
        return self.neuralshape

    def setNeuralStructure(self, neuralshape=np.array([1, 1])):
        # self.setNeuralShape(neuralshape), detta gör en inf-loop
        self.a = []
        self.W = []
        j = 0
        for i in self.neuralshape:
            self.a.append(np.zeros(i))
            self.W.append(np.random.random_sample(np.array([i, j])))
            j = i

    def getNeuralStructure(self):
        return self.a, self.W


def joel():
    matr = np.array([[0, 1, 2, 3],
                     [0, 1, 2, 4]])
    plt.plot(matr[0, :], matr[1, :])
    # plt.show()
    neuralnettest = NeuralNet(np.array([2, 4, 3]))
    try:
        print(str(neuralnettest.a))
        print(str(neuralnettest.W))
        print(str(np.dot(neuralnettest.W[1], neuralnettest.a[0])))
        print(str(neuralnettest.W[1][2, 1]))
    except NameError:
        var_exists = False
    else:
        var_exists = True
    print(var_exists)
