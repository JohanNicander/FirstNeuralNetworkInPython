import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import os
# import abc    används för att göra abstract base class (typ interface)


def sigmoid(x):
    if type(x) is not np.array:
        raise TypeError("Wrong input type to sigmoid")
    return np.divide(1, np.add(1, np.exp(-x)))


class NeuralNet:
    # Variabler:
    # self.neuralShape  vector describing the network,
    #       first element = #input nodes, last element = #output nodes
    # self.a,           list of vectors containing node-values
    # self.W,           list of matricies containing weights
    # self.b,           list of vectors containing biases
    # self.actfun,      activation function

    def __init__(self, neuralShape=np.array([1, 1]), actfun=sigmoid):
        self.setNeuralShape(neuralShape, actfun)

    def setNeuralShape(self, neuralShape=None, actfun=None):
        # Input checks
        if neuralShape is None:
            pass
        elif type(neuralShape) is np.ndarray and neuralShape.ndim == 1:
            self.neuralShape = neuralShape
        else:
            raise ValueError("Argument neuralShape must be a numpy array")

        if actfun is None:
            pass
        elif callable(actfun):
            self.actfun = actfun
        else:
            raise ValueError("argument actfun must be a (callable) function")

        # Initialize a, W, b
        # self.a = []       #Troligen onödig
        self.W = []
        self.b = []
        j = 0
        for i in self.neuralShape:
            # self.a.append(np.zeros(i))       #Troligen onödig
            self.W.append(np.random.random_sample(np.array([i, j])))
            self.b.append(np.random.random(i))
            j = i
        self.W.pop(0)
        self.b.pop(0)

    def propagate(self, input):
        # fattar inte hur man lopar över en lista med np.ndarryobjekt
        # förutsätter att vi inte definerar a

        # Input checks
        # TODO

        for i in range(0, len(self.b)):
            input = self.actfun(np.add(b, np.dot(self.W, input)))
        return input


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
        temp = np.add(1, matr)
    except:
        print("Doesn't work")
    else:
        print(str(temp))
    plt.plot(matr[0, :], matr[1, :])
    # plt.show()
    neuralnettest = NeuralNet(np.array([2, 4, 3]))
    try:
        # print(str(neuralnettest.a))
        print(str(neuralnettest.W))
        print(str(neuralnettest.b))
        # print(str(neuralnettest.actfun(np.array([1, 2, 3]))))
        # print(str(np.add(np.dot(neuralnettest.W[0], neuralnettest.a[0]),
        #                 neuralnettest.b[0])))
        print(str(neuralnettest.W[1][2, 1]))
    except NameError:
        var_exists = False
    else:
        var_exists = True
    print(var_exists)

    a = np.array([[1, 2], [2, 3], [3, 4]])
    print(a.shape)
    tmp = np.array([1, 2])
    if tmp is not np.ndarray:
        print('En array är INTE en array')
        print('En array är en ' + str(type(tmp))
    else:
        print('En array är en array')


joel()
