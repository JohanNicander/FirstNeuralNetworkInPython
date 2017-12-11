import neuralNet as nn
import neuralFuns as nf
import numpy as np


def fun(x):
    return x.prod(axis=0).reshape(1, x.shape[1])


shape = np.array([2, 5, 5, 1])
actfun = [[nf.sigmoid, nf.sigmoidPrime]]
actfun.append([nf.linear, lambda x: np.ones(x.shape)])
net = nn.NeuralNet(shape,)
x = np.array(np.random.random_integers(1, 10, [10000, 2]))
x = x.T
print(x)
y = fun(x)
y.shape = [1, y.size]

print(net.cost(x, y))
res = net.optimCost(x, y, method='Newton-CG', options={'maxiter': 10000})
print(net.getState())
print(net.cost(x, y))

print(res.nit)
print(res.message)
print(res.success)

temp = [np.array([[10], [7]]), np.array([[53], [17]]), np.array([[29], [45]]),
        np.array([[95], [72]])]
for x in temp:
    print(net.error(x, fun(x)))

x = np.random.random_integers(1, 10, [5, 2]).T
print(x)
y = fun(x)
y.shape = [1, y.size]
print(np.mean(net.error(x, y)))
print(net.propagate(x))
