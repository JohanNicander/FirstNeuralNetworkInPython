import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def johan():
    # Read data
    indata = open('Data.txt', 'r')
    dataframe = pd.read_fwf(indata)
    x_values = dataframe[['X']]
    y_values = dataframe[['Y']]

    # Visulize results
    plt.scatter(x_values, y_values)
    plt.show()


class NeuralNet:
    def __init__(self, neuralshape=np.array([1, 1])):
        self.neuralshape = neuralshape
        self.a = []
        self.W = []
        j = 0
        for i in neuralshape:
            self.a.append(np.zeros(i))
            self.W.append(np.zeros(np.array([i, j])))
            j = i


def joel():
    matr = np.array([[0, 1, 2, 3],
                     [0, 1, 2, 4]])
    plt.plot(matr[0, :], matr[1, :])
    # plt.show()
    neuralnettest = NeuralNet(np.array([2, 4, 3]))
    print(str(neuralnettest.a))
    print(str(neuralnettest.W))
    print(str(np.dot(neuralnettest.W[1], neuralnettest.a[0])))
    print(str(neuralnettest.W[1][2, 1]))
