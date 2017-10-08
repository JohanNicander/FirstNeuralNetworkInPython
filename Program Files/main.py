import pandas as pd
import matplotlib.pyplot as plt
import sys

print(sys.path)

# Read data
indata = open('Data.txt', 'r')
dataframe = pd.read_fwf(indata)
x_values = dataframe[['X']]
y_values = dataframe[['Y']]

# Visulize results
plt.scatter(x_values, y_values)
plt.show()


def function(var1, var2):
    print(var1)


class NeuralNet:        # hej
    def __init__(self):
        var = 3
        print(var)
