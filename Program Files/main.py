import numpy as np
import matplotlib.pyplot as pplot


def function(var1, var2):
    print(var1)


class NeuralNet:        # hej
    def __init__(self, strucvec=np.array([1, 1])):
        self.strucvec = strucvec


def plot():
    matr = np.array([[0, 1, 2, 3],
                     [0, 1, 2, 4]])
    pplot.plot(matr[0, :], matr[1, :])
    # pplot.show()
    print(1)


plot()
