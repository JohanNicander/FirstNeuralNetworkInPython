import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import os
# import abc    används för att göra abstract base class (typ interface)


def tempfun(x):
    return x


class NeuralNet:
    # Variabler:
    # self.neuralShape
    # self.a,       list of vectors containing node-values
    # self.W,       list of matricies containing weights
    # self.b,       list of vectors containing biases
    # self.actfun,  aktiveringsfunktion

    def __init__(self, neuralShape=np.array([1, 1]), actfun=tempfun):
        self.setNeuralShape(neuralShape, actfun)

    # Vill ha =self.neuralShape, =self.actfun, som default
    def setNeuralShape(self, neuralShape=None, actfun=None):
        if type(neuralShape) is np.ndarray and neuralShape.ndim == 1:
            self.neuralShape = neuralShape
        else:
            pass
        if callable(actfun):
            self.actfun = actfun
            # TODO: Indikera att något gått fel
        else:
            pass
            # TODO: Indikera att något gått fel
        self.a = []
        self.W = []
        self.b = []
        j = 0
        for i in self.neuralShape:
            self.a.append(np.zeros(i))
            self.W.append(np.random.random_sample(np.array([i, j])))
            self.b.append(np.random.random(i))
            j = i
        self.W.pop(0)
        self.b.pop(0)


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


def joel():
    matr = np.array([[0, 1, 2, 3],
                     [0, 1, 2, 4]])
    try:
        temp = np.add(matr, matr)
    except:
        print("Doesn't work")
    else:
        print(str(temp))
    plt.plot(matr[0, :], matr[1, :])
    # plt.show()
    neuralnettest = NeuralNet(np.array([2, 4, 3]))
    try:
        print(str(neuralnettest.a))
        print(str(neuralnettest.W))
        print(str(neuralnettest.b))
        print(str(neuralnettest.actfun(np.array([1, 2, 3]))))
        print(str(np.add(np.dot(neuralnettest.W[0], neuralnettest.a[0]),
                         neuralnettest.b[0])))
        print(str(neuralnettest.W[1][2, 1]))
    except NameError:
        var_exists = False
    else:
        var_exists = True
    print(var_exists)


joel()
