import neuralNet as nn
import neuralFuns as nf
import matplotlib.pyplot as plt
import numpy as np
import math


def fun(x):
    return 10 * np.sin(x)


plotx = np.linspace(start=-6 * math.pi, stop=10 * math.pi, num=1000).\
    reshape([1, 1000])
ploty = fun(plotx)
plt.plot(plotx.T, ploty.T, c='r')
# plt.show()
# quit()

shape = np.array([1, 5, 5, 5, 5, 1])
actfun = [[nf.sigmoid, nf.sigmoidPrime],
          [nf.linear, lambda x: np.ones(x.shape)]]
net = nn.NeuralNet(shape)
x = np.linspace(start=-4 * math.pi, stop=4 * math.pi, num=1000).\
    reshape([1, 1000])
y = fun(x)
y.shape = [1, y.size]

######
grad = net.getState(*net.gradError(x, y))
numgrad = net.gradErrorNumerical(x, y)
print(np.linalg.norm(grad))
print(np.linalg.norm(numgrad))
print(np.linalg.norm(grad - numgrad))

# quit()
# ####

# BFGS, SLSQP
print(net.cost(x, y))
res = net.optimCost(x, y, setStateRegardless=True, method='SLSQP',
                    options={'maxiter': 10**5, 'ftol': 10**-100})
# print(net.getState())
print(net.cost(x, y))
print(net.cost(plotx, ploty))

print(res.nit)
print(res.message)
print(res.success)
print(net.gradCost(x, y))


plt.plot(plotx.T, net.propagate(plotx).T, c='b')
plt.show()
