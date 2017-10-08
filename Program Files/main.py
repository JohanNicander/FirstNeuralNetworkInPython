<<<<<<< HEAD
import pandas as pd
import matplotlib.pyplot as plt
import sys


# Read data
indata = open('Data.txt', 'r')
dataframe = pd.read_fwf(indata)
x_values = dataframe[['X']]
y_values = dataframe[['Y']]

# Visulize results
plt.scatter(x_values, y_values)
plt.show()
=======
import numpy as np
import matplotlib.pyplot as plt
>>>>>>> master


def function(var1, var2):
    print(var1)


class NeuralNet:        # hej
    def __init__(self, strucvec=np.array([1, 1])):
        self.strucvec = strucvec


<<<<<<< HEAD
def plot():
    matr = np.array([[0, 1, 2, 3],
                     [0, 1, 2, 4]])
    pplot.plot(matr[0, :], matr[1, :])
    # pplot.show()
    print(1)


plot()
=======
matr = np.array([[0, 1, 2, 3],
                 [0, 1, 2, 4]])
plt.plot(matr[0, :], matr[1, :])
plt.show()
>>>>>>> master
