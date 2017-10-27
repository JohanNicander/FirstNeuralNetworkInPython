import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import os
import sys
import io
# import abc    används för att göra abstract base class (typ interface)

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')


def sigmoid(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to sigmoid")
    return np.divide(1, np.add(1, np.exp(-x)))


class NeuralNet:
    # Variables:
    # self.neuralShape  vector describing the network,
    #       first element = #input nodes, last element = #output nodes
    # self.a,           list of vectors containing node-values after activation
    # self.z,           list if vectors containing node-values befor activation
    # self.d            list of vectors containing node-errors
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
        self.a = []
        self.z = []
        self.d = []
        self.W = []
        self.b = []
        j = 0
        for i in self.neuralShape:
            self.a.append(np.zeros(i))
            self.d.append(np.zeros(i))
            self.z.append(np.zeros(i))
            self.W.append(np.random.random_sample(np.array([i, j])))
            self.b.append(np.random.random(i))
            j = i
        self.W.pop(0)
        self.b.pop(0)

    def propagate(self, x):
        # Input checks
        if x.shape[0] != self.neuralShape[0]:
            raise ValueError("x.shape[0] must equal neuralShape[0]")
        if x.ndim == 1:
            x.shape = [len(x), 1]
        B = []
        for b in self.b:
            B.append(np.multiply(b, np.ones([b.shape[0], x.shape[1]])))
        self.a[0] = x
        for i in range(0, len(self.b) - 2):
            self.z[i + 1] = np.add(B[i], np.dot(self.W[i], self.a[i]))
            self.a[i + 1] = self.actfun(self.z[i + 1])


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
    except Exception as e:
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
        print(str(neuralnettest.actfun(np.array([1, 2, 3]))))
        # print(str(np.add(np.dot(neuralnettest.W[0], neuralnettest.a[0]),
        #                 neuralnettest.b[0])))
        print(str(neuralnettest.W[1][2, 1]))
    except NameError:
        var_exists = False
    else:
        var_exists = True
    print(var_exists)

    a = np.array([[[1, 2], [2, 3], [3, 4]], [[1, 2], [2, 3], [3, 4]],
                  [[1, 2], [2, 3], [3, 4]], [[1, 2], [2, 3], [3, 4]]])
    print(a.shape)

    b = np.ones([4, 3])
    c = np.array([1, 2, 3, 4])
    c.shape = [4, 1]
    print(str(np.multiply(c, b)))
    print(str(np.multiply(c, b).shape))


joel()
