import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
import os
import sys
import io
import time         # används för tidsmätning
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

# TODO vi borde kanske göra en egen modul med aktiveringsfunktionerna ovan,
#       för att denna fil inte ska bli för lång. Borde även flytta johan()
#       och joel() till egna filer (där vi kan importera main.py) för att inte
#       skräpa ner så mycket här

class NeuralNet:
    # Variables:
    # self.neuralShape  vector describing the network,
    #       first element = #input nodes, last element = #output nodes
    # self.W,           list of matricies containing weights
    # self.b,           list of vectors containing biases
    # self.actfun,      activation functions and derivatives
    # self.actfun[0][0] hidden layer activation function
    # self.actfun[0][1] hidden layer activation function derivative
    # self.actfun[1][0] output layer activation function
    # self.actfun[1][1] output layer activation function derivative
    # Variables existing during propagation and training:
    # self.a,           list of vectors containing node-values after activation
    # self.z,           list if vectors containing node-values befor activation

    # TODO det verkar som att np blir lite snabbare om man använder endim-
    # arrayer (istället för tvådim kolonnvektorer)

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
            self.N = neuralShape.shape[0]
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
            self.b.append(np.random.random_sample(np.array([i, 1])))
            j = i
        self.W.pop(0)
        self.b.pop(0)

    def propagate(self, x):
        # Input checks
        if type(x) is not np.ndarray or x.ndim > 2:
            raise ValueError("x must be a numpyarray of dimension at most 2")
        elif x.shape[0] != self.neuralShape[0]:
            raise ValueError("x.shape[0] must equal neuralShape[0]")

        if x.ndim == 1:
            x.shape = [len(x), 1]
        self.z = [None]
        self.a = [x]
        for i in range(0, self.N - 1):
            self.z.append = np.add(self.b[i], np.dot(self.W[i], self.a[i]))
            self.a.append = self.actfun(self.z[i + 1])

    def gradient(self, x, y):
        self.propagate(x)
        if type(y) is not np.ndarray or y.shape != self.a[-1].shape:
            raise ValueError("y must be a numpy array of shape \
                              [neuralShape[-1], x.shape[1]]")

        # M = self.a[0].shape[1]
        O = np.ones([self.a[0].shape[1], 1])    # O = np.ones([M, 1])
        d = [np.multiply(np.subtract(self.a[-1], y),
                         self.actfun[1][1](self.z[-1]))]
        dJdW = [np.dot(d[0], self.a[-2].T)]
        dJdb = [np.dot(d[0], O)]
        for i in range(2, self.N):
            d.insert(0, np.multiply(np.dot(self.W[-i + 1].T, d[-i + 1]),
                                    self.actfun[0][1](self.z[-i])))
            dJdW.insert(0, np.dot(d[0], self.a[-i - 1].T))
            dJdb.insert(0, np.dot(d[0], O))
        return dJdW, dJdb

    def train(self, x, y):
        if type(x) and type(y) is not np.ndarray:
            raise ValueError("Arguments must be numpy arrays")
        elif x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have same number of columns")

        d = []
        dJdW = []
        for i in range(0, len(self.neuralShape)):
            d[i] = np.zeros(self.a[i].shape)
            dJdW[i] = np.zeros(self.W[i - 1].shape)
        dJdW.pop(0)


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
    #
    # matr = np.array([[0, 1, 2, 3],
    #                  [0, 1, 2, 4]])
    # try:
    #     temp = np.add(1, matr)
    # except Exception as e:
    #     print("Doesn't work")
    # else:
    #     print(str(temp))
    # plt.plot(matr[0, :], matr[1, :])
    # # plt.show()
    # neuralnettest = NeuralNet(np.array([2, 4, 3]))
    # try:
    #     # print(str(neuralnettest.a))
    #     print(str(neuralnettest.W))
    #     print(str(neuralnettest.b))
    #     # print(str(neuralnettest.actfun(np.array([1, 2, 3]))))
    #     # print(str(np.add(np.dot(neuralnettest.W[0], neuralnettest.a[0]),
    #     #                 neuralnettest.b[0])))
    #     print(str(neuralnettest.W[1][2, 1]))
    # except NameError:
    #     var_exists = False
    # else:
    #     var_exists = True
    # print(var_exists)
    #
    # a = np.array([[1, 2], [2, 3], [3, 4]])
    # print(a.shape)
    #
    # b = np.ones([4, 3])
    # c = np.array([1, 2, 3, 4])
    # c.shape = [4, 1]
    # print(str(np.add(c, b)))
    # print(str(np.add(c, b).shape))
    #
    # d = np.array([[1, -2, 0, -3], [-6, 9, 2, -5]])
    # print(str(reLU(d)))
    # print(str(reLUPrime(d)))
    #
    # e = np.sum(np.array([[1, 2], [3, 4]]), 1)
    # e.shape = [e.shape[0], 1]
    # print(str(e))
    # print(str(e.shape))
    #
    # f = [(1, 2), (3, 4)]
    # print(len(f))
    # print(str(f[0][1]))
    #
    # actfun = [[sigmoid, sigmoidPrime], [sigmoid, sigmoidPrime]]
    # print(len(actfun))
    # print(len(actfun[0]))
    # print(len(actfun[1]))
    # print(str(actfun[0][1](np.array([1]))) + ' ' +
    #       str(actfun[1][0](np.array([1]))))
    # if len(actfun) and len(actfun[0]) and len([1]) == 2:
    #     print('seems NOT to work')
    #
    # g = []
    # for i in range(1, 5):
    #     g.insert(0, i)
    # print(g)
    #
    # A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # x = np.array([9, 5, 3])
    # b1 = np.dot(A, x)
    # b2 = np.dot(x, A)
    # b3 = np.dot(A, np.reshape(x, [3, 1]))
    # print(b1)
    # print(b2)
    # print(b3)
    # x2 = np.reshape(x, [3, 1])
    # start = time.time()
    # for i in range(1000000):
    #     np.dot(A, x2)
    # end = time.time()
    # print('Andra tog ' + str(end - start) + ' s att beräkna')
    # start = time.time()
    # for i in range(1000000):
    #     np.dot(A, x)
    # end = time.time()
    # print('Första tog ' + str(end - start) + ' s att beräkna')
    #
    # for j in range(100):
    #     A1 = np.random.random_sample([40, 50])
    #     x1 = np.random.random_sample([50, 1000])
    #     A2 = A1.T
    #     x2 = x1.T
    #     l1 = []
    #     l2 = []
    #
    #     start1 = time.time()
    #     for i in range(10000):
    #         np.dot(A1, x1)
    #     end1 = time.time()
    #     l1.append(end1 - start1)
    #
    #     start2 = time.time()
    #     for i in range(10000):
    #         np.dot(x2, A2)
    #     end2 = time.time()
    #     l2.append(end2 - start2)
    #     print(sum(l1) / len(l1))
    #     print(sum(l2) / len(l2))
    #     print(j)
    #     sys.stdout.flush()
    a = [np.array([1, 2])]
    for i in range(0, 9):
        a.append(np.array([i]))
    print(a)
    pass


joel()
