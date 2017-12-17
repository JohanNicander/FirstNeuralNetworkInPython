import numpy as np
import sys
import io
import os
import neuralFuns as nf
from scipy import optimize
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# TODO, update cost function with a lambda*complexity term
#       and derive new gradient. Optimize for lambda? Golden ratio search?
# TODO, stochastic gradient descent use a subset of training data for each step
# Should cost and error be avarages of cost/error per training example?

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

    # TODO, should there be two complexity factors?
    #           k1 \in [0, inf), used directly in cost
    #           k2 \in (0, 1],  might be better to for golden ratio search (?)

    def __init__(self, neuralShape, state=None, actfun=None, compfun=None,
                 compfact=0):
        self.setNeuralShape(neuralShape)
        self.W = [None] * (neuralShape.size - 1)
        self.b = [None] * (neuralShape.size - 1)
        if state is None:
            state = np.random.random_sample(np.dot(neuralShape[:-1] + 1,
                                                   neuralShape[1:]))
        self.setState(state)

        if actfun is None:
            actfun = [[nf.reLU, nf.reLUPrime]]
            if self.neuralShape[-1] == 1:
                actfun.append([nf.linear, lambda x: np.ones(x.shape)])
            else:
                actfun.append([nf.softmax, nf.softmaxPrime])
        self.setActFun(actfun)

        if compfun is None:
            compfun = [lambda x: 0, lambda x: np.zeros(x.shape)]
        self.setCompFun(compfun)
        self.setCompFact(compfact)

    def __str__(self):
        # TODO: can't we do better than this?
        return str(self.getState())

    def __repr__(self):
        # s = ("NeuralNet(neuralShape=np.%r, state=np.%r actfun=nf.%s,"
        #      " compfun=nf.%r, compfact=%r)")
        # out = s % (self.neuralShape, self.getState(), self.actfun,
        #            self.compfun, self.compfact)
        out = "NeuralNet(neuralShape=np.%r, state=np.%r)" \
              % (self.neuralShape, self.getState())
        return out

###############################################################################
# Setters and getters
###############################################################################

    def setNeuralShape(self, neuralShape, W=None, b=None):
        # TODO: try converting to ndarray
        # TODO: update weights and biases for consistency
        if type(neuralShape) is not np.ndarray or neuralShape.ndim != 1:
            raise ValueError("Argument neuralShape must be a numpy array")
        else:
            self.neuralShape = neuralShape

    def setActFun(self, actfun):
        if type(actfun) is not list and not tuple:
            raise TypeError("Argument actfun must be a list or a tuple")
        elif any(len(l) != 2 for l in actfun) or len(actfun) != 2:
            raise ValueError("Argument actfun must have shape 2x2")
        elif any(not callable(f) for f in [item for sublist in
                                           actfun for item in sublist]):
            raise ValueError("Each element must be a (callable) function")
        else:
            self.actfun = actfun

    def setCompFun(self, compfun):
        if type(compfun) is not list and not tuple:
            raise TypeError("Argument compfun must be a list or a tuple")
        elif len(compfun) != 2:
            raise ValueError("Argument compfun must be of length 2")
        elif any(not callable(f) for f in compfun):
            raise ValueError("Each element must be a (callable) function")
        else:
            self.compfun = compfun

# TODO: fix better checks here
    def setCompFact(self, compfact):
        try:
            np.multiply(compfact, np.ones([3, 2]))
            np.multiply(compfact, np.ones([4, 7]))
        except:
            raise TypeError("compfact should work with np.multiply")
        self.compfact = compfact

    def setState(self, wlist):
        for i in range(self.neuralShape.size - 1):
            self.W[i] = wlist[:self.neuralShape[i + 1]
                              * self.neuralShape[i]].reshape(
                [self.neuralShape[i + 1], self.neuralShape[i]])
            wlist = wlist[self.neuralShape[i + 1] * self.neuralShape[i]:]
            self.b[i] = wlist[:self.neuralShape[i + 1]].reshape(
                self.neuralShape[i + 1], 1)
            wlist = wlist[self.neuralShape[i + 1]:]

# Another way to do things, which is better/faster?
    def getState(self, W=None, b=None):
        if W is None:
            W = self.W
        if b is None:
            b = self.b
        temp = []
        for i in range(len(W)):
            temp.extend(np.ndarray.tolist(W[i].ravel()))
            temp.extend(np.ndarray.tolist(b[i].ravel()))
        return np.array(temp)

    def save(self, location=None):
        if location is None:
            Tk().withdraw()
            location = asksaveasfilename(defaultextension=".nn",
                                         initialdir=os.path.dirname(
                                             os.path.abspath(__file__)),
                                         filetypes=(("Neural Net File",
                                                     "*.nn"),
                                                    ("All Files", "*.*")))
        with open(location, 'w+') as f:
            f.write(repr(self))
        f.close()

###############################################################################
# General NeuralNet functions
###############################################################################

    def propagate(self, x):
        # Input checks
        # TODO: try converting to ndarray (?)
        if type(x) is not np.ndarray or x.ndim > 2:
            raise ValueError("x must be a numpyarray of dimension at most 2")
        elif x.shape[0] != self.neuralShape[0]:
            raise ValueError("x.shape[0] must equal neuralShape[0]")

        if x.ndim == 1:
            x.shape = [x.size, 1]
        self.z = [None]
        self.a = [x]
        length = self.neuralShape.size

        def propagateInner(fun, i):
            self.z.append(np.add(self.b[i], np.dot(self.W[i], self.a[i])))
            self.a.append(fun(self.z[i + 1]))

        for j in range(length - 2):
            propagateInner(self.actfun[0][0], j)
        propagateInner(self.actfun[1][0], length - 2)

        return self.a[-1]

    def error(self, x, y):
        nrex = x.shape[1]
        self.propagate(x)
        if type(y) is not np.ndarray or y.shape != self.a[-1].shape:
            raise ValueError("y must be a numpy array of shape " +
                             str([self.neuralShape[-1], x.shape[1]]))
        return np.multiply(np.divide(1, 2 * nrex),
                           np.sum(np.square(np.subtract(self.a[-1], y))))

    def cost(self, x, y):
        complexity = 0
        for i in self.W:
            complexity += self.compfun[0](i)
        return np.add(self.error(x, y), np.multiply(self.compfact, complexity))

    def gradError(self, x, y):
        self.propagate(x)
        nrex = x.shape[1]
        if type(y) is not np.ndarray or y.shape != self.a[-1].shape:
            raise ValueError("y must be a numpy array of shape" +
                             str([self.neuralShape[-1], x.shape[1]]))

        # M = self.a[0].shape[1]
        O = np.ones([self.a[0].shape[1], 1])    # O = np.ones([M, 1])
        d = [np.multiply(np.subtract(self.a[-1], y),
                         self.actfun[1][1](self.z[-1]))]
        dJdW = [np.dot(d[0], self.a[-2].T) / nrex]
        dJdb = [np.dot(d[0], O) / nrex]
        for i in range(2, self.neuralShape.size):
            d.insert(0, np.multiply(np.dot(self.W[-i + 1].T, d[-i + 1]),
                                    self.actfun[0][1](self.z[-i])))
            dJdW.insert(0, np.dot(d[0], self.a[-i - 1].T) / nrex)
            dJdb.insert(0, np.dot(d[0], O) / nrex)
        return dJdW, dJdb

    def gradCost(self, x, y):
        # might be a good idea to eventually remove gradError (since it should
        # not be used) and do all calculations here instead
        return self.gradError(x, y)     # TODO: add complexity term

    def gradErrorNumerical(self, x, y):
        state = self.getState()
        numgrad = np.zeros(state.shape)
        perturb = np.zeros(state.shape)
        e = 1e-6

        for p in range(state.size):
            # Set perturbation vector
            perturb[p] = e
            self.setState(state + perturb)
            loss2 = self.cost(x, y)

            self.setState(state - perturb)
            loss1 = self.cost(x, y)

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2 * e)

            # Return the value we changed to zero:
            perturb[p] = 0

        # Return Params to original value:
        self.setState(state)

        return numgrad

###############################################################################
# Training and optimizing
###############################################################################

    def optimWrapper(self, state, x, y):
        self.setState(state)
        temp = self.gradCost(x, y)
        return [self.cost(x, y), self.getState(*temp)]

    def optimCost(self, x, y, setStateRegardless=False, **kwargs):
        # setx: sets W and b regardless of success
        if type(x) and type(y) is not np.ndarray:
            raise ValueError("Arguments must be numpy arrays")
        elif x.shape[1] != y.shape[1]:
            raise ValueError("x and y must have same number of columns")

        # TODO: can minimize take options as kwargs?? No...
        # TODO: better handling of options...
        defaultoptions = {'maxiter': 200, 'disp': False}
        tempdict = {'fun': self.optimWrapper, 'x0': self.getState(),
                    'args': (x, y), 'method': 'BFGS', 'jac': True,
                    'options': defaultoptions}
        for key, default in tempdict.items():
            if key not in kwargs:
                # TODO: checks?
                kwargs[key] = default
        temp = self.getState()
        optimRes = optimize.minimize(**kwargs)
        if optimRes.success or setStateRegardless:
            self.setState(optimRes.x)
        else:
            self.setState(temp)
        return optimRes

    def train(self):
        pass

###############################################################################
# Outside class
###############################################################################


def load(location=None):
    if location is None:
        Tk().withdraw()
        location = asksaveasfilename(defaultextension=".nn",
                                     initialdir=os.path.dirname(
                                         os.path.abspath(__file__)),
                                     filetypes=(("Neural Net File",
                                                 "*.nn"),
                                                ("All Files", "*.*")))
    with open(location, 'r') as f:
        read_data = f.read()
    f.close()
    return eval(read_data)
