import numpy as np
import matplotlib.pyplot as plt


def function(var1, var2):
    print(var1)


class NeuralNet:        # hej
    def __init__(self, strucvec=np.array([1, 1])):
        self.strucvec = strucvec


def plot():
    matr = np.array([[0, 1, 2, 3],
                     [0, 1, 2, 4]])
    plt.plot(matr[0, range(4)], matr[1, range(4)])
    plt.show()
    print(1)


plot()
