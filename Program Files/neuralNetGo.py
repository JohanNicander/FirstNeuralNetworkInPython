import neuralNet as nn
import numpy as np


shape = np.array([3, 3, 2, 2])
net = nn.NeuralNet(shape)
x = np.array([1, 3, 4])
x.shape = [3, 1]
y = np.array([1, 0])
y.shape = [2, 1]
print(net.optimCost(x, y))
