import neuralNet as nn
import numpy as np


def fun(x):
    return x.prod(axis=0)


shape = np.array([2, 3, 3, 1])
net = nn.NeuralNet(shape)
x = np.array([[1, 3], [2, 4], [10, 7], [13, 5]])
x = x.T
y = fun(x)
y.shape = [1, y.size]

print(net.gradCost(x, y))
print(net.getState())
print(net.cost(x, y))
net.optimCost(x, y)
print(net.getState())
print(net.cost(x, y))
