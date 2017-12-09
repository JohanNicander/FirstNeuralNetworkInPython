import neuralNet as nn
import numpy as np


def fun(x):
    return x.prod(axis=0).reshape(1, x.shape[1])


shape = np.array([2, 5, 5, 5, 1])
net = nn.NeuralNet(shape)
x = 20 * np.random.random_sample([10000, 2])
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

temp = [np.array([[10], [7]]), np.array([[53], [17]]), np.array([[29], [45]]),
        np.array([[95], [72]])]
for x in temp:
    print(net.error(x, fun(x)))
