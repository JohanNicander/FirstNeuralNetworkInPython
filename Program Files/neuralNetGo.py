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

# print(net.getState())
# print(net.cost(x, y))
#
# temp = net.gradCost(x, y)
# print(temp)
# net.setState(net.getState() - 0.001 * net.getState(*temp))
# print(net.getState())
# print(net.cost(x, y))


print(net.getState())
print(net.cost(x, y))
res = net.optimCost(x, y, method='BFGS', options={'maxiter': 10000})
print(net.getState())
print(net.cost(x, y))

print(res.nit)
print(res.message)
print(res.success)

print("hej")

print(net.propagate(np.array([10, 7])))
