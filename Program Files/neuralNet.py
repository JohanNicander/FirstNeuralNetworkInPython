import numpy as np
import sys
import io
import neuralFuns as nf

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# TODO, update cost function with a lambda*complexity term
#       and derive new gradient. Optimize for lambda? Golden ratio search?
# TODO, stochastic gradient descent use a subset of training data for each step
# Should cost and error be avarages of cost/error per training example?
# TODO, setters for actfun and compfun (complexity function)

# TODO, improve initialization:
# the recommended heuristic is to initialize each neuron’s weight vector as:
# w = np.random.randn(n) / sqrt(n), where n is the number of its inputs.
# This ensures that all neurons in the network initially have approximately the
# same output distribution and empirically improves the rate of convergence.
# In practice, the current recommendation is to use ReLU units and use the
# w = np.random.randn(n) * sqrt(2.0/n)
# Initializing the biases:
# It is possible and common to initialize the biases
# to be zero, since the asymmetry breaking is provided by the small random
# numbers in the weights. For ReLU non-linearities, some people like to use
# small constant value such as 0.01 for all biases because this ensures that
# all ReLU units fire in the beginning and therefore obtain and propagate some
# gradient. However, it is not clear if this provides a consistent improvement
# (in fact some results seem to indicate that this performs worse) and it is
# more common to simply use 0 bias initialization.
# For more info, see: http://cs231n.github.io/neural-networks-2/


class NeuralNet:
    # Network parameters:
    # self.W,           list of matricies containing weights
    # self.b,           list of vectors containing biases
    #
    # Propagation parameters:
    # self.a,           list of vectors containing node-values after activation
    # self.z,           list if vectors containing node-values befor activation
    #
    # Hyper parameters:
    # self.neuralShape  vector describing the network,
    #       first element = #input nodes, last element = #output nodes
    # self.k            complexity factor, C = J + k*compfun
    #
    # Network functions and hyper functions:
    # self.actfun,      activation functions and derivatives
    #   self.actfun[0][0]       hidden layer activation function
    #   self.actfun[0][1]       hidden layer activation function derivative
    #   self.actfun[1][0]       output layer activation function
    #   self.actfun[1][1]       output layer activation function derivative
    # self.compfun,     complexity function and derivative
    #   self.compfun[0]     complexity function
    #   self.compfun[1]     complexity function derivative

    def __init__(self, neuralShape=np.array([1, 1]),
                 actfun=[[nf.sigmoid, nf.sigmoidPrime],
                         [nf.sigmoid, nf.sigmoidPrime]]):
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
        return self.a[-1]

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
