import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def johan():
    # Read data
    indata = open('Data.txt', 'r')
    dataframe = pd.read_fwf(indata)
    x_values = dataframe[['X']]
    y_values = dataframe[['Y']]

    # Visulize results
    plt.scatter(x_values, y_values)
    plt.show()


class NeuralNet:
    def __init__(self, strucvec=np.array([1, 1])):
        self.strucvec = strucvec


def joel():
    matr = np.array([[0, 1, 2, 3],
                     [0, 1, 2, 4]])
    plt.plot(matr[0, :], matr[1, :])
    plt.show()
