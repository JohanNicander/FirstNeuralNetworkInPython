import neuralNet as nn
import numpy as np


def fun(x):
    return x.prod(axis=0)


shape = np.array([2, 3, 3, 1])
net = nn.NeuralNet(shape)
x = 100 * np.random.random_sample([1000, 2])
print(x)
x = x.T
y = fun(x)
y.shape = [1, y.size]

print(net.cost(x, y))
res = net.optimCost(x, y, setx=True, method='BFGS', options={'maxiter': 10000})
print(net.getState())
print(net.cost(x, y))

print(res.nit)
print(res.message)
print(res.success)

print(net.propagate(np.array([10, 7])))
