import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import os
import numpy as np
# Read data
# path = os.path.dirname(os.path.abspath(__file__))
# indata = open(path + r'/Data.txt')
# dataframe = pd.read_fwf(indata)
# x_values = dataframe[['X']]
# y_values = dataframe[['Y']]
# print(x_values)
# # Liear regression
# reg = linear_model.LinearRegression()
# reg.fit(x_values, y_values)
# # Visulize results
# plt.scatter(x_values, y_values)
# plt.plot(x_values, reg.predict(x_values))
# plt.show()
x = 5
y = 7
if x and y is not None:
    print(x + y)
