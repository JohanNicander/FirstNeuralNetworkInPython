import numpy as np
import sys
import io
import actfuns as af

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# TODO, update cost function with a lambda*complexity term
#       and derive new gradient. Optimize for lambda? Golden ratio search?


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

    def __init__(self, neuralShape=np.array([1, 1]),
                 actfun=[[af.sigmoid, af.sigmoidPrime],
                         [af.sigmoid, af.sigmoidPrime]]):
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

    def error(self, x, y):
        self.propagate(x)
        if type(y) is not np.ndarray or y.shape != self.a[-1].shape:
            raise ValueError("y must be a numpy array of shape \
                              [neuralShape[-1], x.shape[1]]")
        return np.multiply(np.divide(1, 2),
                           np.sum(np.power(np.subtract(self.a[-1], y), 2)))

    def cost(self, x, y, k):
        error = self.error(x, y)
        complexity = []  # TODO
        return error + k * complexity

    def gradCost(self, x, y):       # TODO, add complexity term
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
