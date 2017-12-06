import neuralNet as nn
import numpy as np


shape = np.array([6, 6, 3, 1])
net = nn.NeuralNet(shape)
temp = net.getState3()

print(temp)
