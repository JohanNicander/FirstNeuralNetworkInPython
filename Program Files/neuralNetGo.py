import neuralNet as nn
import numpy as np


shape = np.array([3, 3, 2, 2])
net = nn.NeuralNet(shape)
x = np.array([1, 3, 4])
net.propagate(x)
