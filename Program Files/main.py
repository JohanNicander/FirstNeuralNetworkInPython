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

# \\TODO: ändra till tuplar


def sigmoid(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to sigmoid")
    return np.divide(1, np.add(1, np.exp(-x)))


def sigmoidPrime(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to sigmoidPrime")
    return np.divide(np.exp(x), np.square(np.add(1, np.exp(x))))


def reLU(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to reLU")
    return np.maximum(0, x)


def reLUPrime(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to reLUPrime")
    return np.heaviside(x, 0.5)


def softmax(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to softmax")
    return np.divide(np.exp(x), np.sum(np.exp(x), 0))
# //TODO: Kontrolera att summering är över rätt axel
# //TODO: kan kanske villja göra något med -max eller nått för inte inf


def softmaxPrime(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to softmaxPRIME")
    return np.multiply(np.squre(softmax(x)), np.devide(np.sum(np.exp(x), 0),
                                                       np.exp(x)) - 1)


def linear(x, a=1):
    if type(x) is not np.ndarray or type(a) is not np.ndarray:
        raise TypeError("Wrong input type to linear")
    return np.multiply(np.sum(x, 0), a)  # //TODO: Dimentioner och så...


# def linearPRIME(a=1):
#    if type(a) is not np.ndarray:
#        raise TypeError("Wrong input type to linearPRIME")
#    return a  # //TODO: ÖM... Ja..


class NeuralNet:
    # Variables:
    # self.neuralShape  vector describing the network,
    #       first element = #input nodes, last element = #output nodes
    # self.a,           list of vectors containing node-values after activation
    # self.z,           list if vectors containing node-values befor activation
    # self.W,           list of matricies containing weights
    # self.b,           list of vectors containing biases
    # self.actfun,      activation function

    def __init__(self, neuralShape=np.array([1, 1]), actfun=[[sigmoid,
                                                              sigmoidPrime],
                                                             [sigmoid,
                                                              sigmoidPrime]]):
        if neuralShape and actfun is not None:
            self.setNeuralShape(neuralShape, actfun)

    def setNeuralShape(self, neuralShape=None, actfun=None):
        if neuralShape is None:
            pass
        elif type(neuralShape) is np.ndarray and neuralShape.ndim == 1:
            self.neuralShape = neuralShape
        else:
            raise ValueError("Argument neuralShape must be a numpy array")

        if actfun is None:
            pass
        elif (len(actfun) and len(actfun[0]) and len(actfun[1]) == 2) and \
                callable(actfun[0][0]) and callable(actfun[0][1]) and \
                callable(actfun[1][0]) and callable(actfun[1][1]):
            self.actfun = actfun
        else:
            raise ValueError("Argument actfun must have shape 2x2 and each \
                              element must be a (callable) function")

        # Initialize W, b
        self.W = []
        self.b = []
        j = 0
        for i in self.neuralShape:
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
        self.a[0] = x
        for i in range(0, len(self.b) - 2):
            self.z[i + 1] = np.add(self.b[i], np.dot(self.W[i], self.a[i]))
            self.a[i + 1] = self.actfun(self.z[i + 1])

    def gradient(self, x, y):
        self.propagate(x)

        # //TODO: Flytta till init train
        d = []
        dJdW = []
        for i in range(0, len(self.neuralShape)):
            d[i] = np.zeros(self.a[i].shape)
            dJdW[i] = np.zeros(self.W[i - 1].shape)
        dJdW.pop(0)

        d[-1] = np.multiply(self.a[-1] - y,
                            self.actfun[-1](self.z[-1]))
        for i in range(1, len(self.neuralShape)):
            dJdW[-i] = np.dot(self.a[-i - 1].T, d[-i])
            d[-i] = np.dot(d[-i - 1], self.W[-i].T) * \
                self.actfun[-i - 1](self.z[-i - 1])
            # \\TODO: Kolla index

        return dJdW, dJdb

    def train(self, x, y):
        if type(x) and type(y) is not np.ndarray:
            raise ValueError("Arguments must be numpy arrays")
        elif x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have same number of columns")


def johan():
    # Read data
    # path = os.path.dirname(os.path.abspath(__file__))
    # indata = open(path + r'/Data.txt')
    # dataframe = pd.read_fwf(indata)
    # x_values = dataframe[['X']]
    # y_values = dataframe[['Y']]
    # print(x_values)
    # # Liear regression
    # reg = linear_model.LinearRegression()
    # reg.fit(x_values, y_values)
    # # Visulize results
    # plt.scatter(x_values, y_values)
    # plt.plot(x_values, reg.predict(x_values))
    # plt.show()
    x = 5
    y = 7
    if x and y is not None:
        print(x + y)


def joel():
    # '''
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

    b = np.ones([4, 3])
    c = np.array([1, 2, 3, 4])
    c.shape = [4, 1]
    print(str(np.add(c, b)))
    print(str(np.add(c, b).shape))

    d = np.array([[1, -2, 0, -3], [-6, 9, 2, -5]])
    print(str(reLU(d)))
    print(str(reLUPrime(d)))

    e = np.sum(np.array([[1, 2], [3, 4]]), 1)
    e.shape = [e.shape[0], 1]
    print(str(e))
    print(str(e.shape))

    f = [(1, 2), (3, 4)]
    print(len(f))
    print(str(f[0][1]))
    # '''
    actfun = [[sigmoid, sigmoidPrime], [sigmoid, sigmoidPrime]]
    print(len(actfun))
    print(len(actfun[0]))
    print(len(actfun[1]))
    print(str(actfun[0][1](np.array([1]))) + ' ' +
          str(actfun[1][0](np.array([1]))))
    if len(actfun) and len(actfun[0]) and len([1]) == 2:
        print('seems NOT to work')


joel()
