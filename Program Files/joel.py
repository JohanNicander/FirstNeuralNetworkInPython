import numpy as np
import neuralNet as nn
import neuralFuns as nf
import time         # används för tidsmätning
# import abc    används för att göra abstract base class (typ interface)

time.time()
# nn.NeuralNet()

# matr = np.array([[0, 1, 2, 3],
#                  [0, 1, 2, 4]])
# try:
#     temp = np.add(1, matr)
# except Exception as e:
#     print("Doesn't work")
# else:
#     print(str(temp))
# plt.plot(matr[0, :], matr[1, :])
# # plt.show()
# neuralnettest = NeuralNet(np.array([2, 4, 3]))
# try:
#     # print(str(neuralnettest.a))
#     print(str(neuralnettest.W))
#     print(str(neuralnettest.b))
#     # print(str(neuralnettest.actfun(np.array([1, 2, 3]))))
#     # print(str(np.add(np.dot(neuralnettest.W[0], neuralnettest.a[0]),
#     #                 neuralnettest.b[0])))
#     print(str(neuralnettest.W[1][2, 1]))
# except NameError:
#     var_exists = False
# else:
#     var_exists = True
# print(var_exists)
#
# a = np.array([[1, 2], [2, 3], [3, 4]])
# print(a.shape)
#
# b = np.ones([4, 3])
# c = np.array([1, 2, 3, 4])
# c.shape = [4, 1]
# print(str(np.add(c, b)))
# print(str(np.add(c, b).shape))
#
# d = np.array([[1, -2, 0, -3], [-6, 9, 2, -5]])
# print(str(reLU(d)))
# print(str(reLUPrime(d)))
#
# e = np.sum(np.array([[1, 2], [3, 4]]), 1)
# e.shape = [e.shape[0], 1]
# print(str(e))
# print(str(e.shape))
#
# f = [(1, 2), (3, 4)]
# print(len(f))
# print(str(f[0][1]))
#
# actfun = [[sigmoid, sigmoidPrime], [sigmoid, sigmoidPrime]]
# print(len(actfun))
# print(len(actfun[0]))
# print(len(actfun[1]))
# print(str(actfun[0][1](np.array([1]))) + ' ' +
#       str(actfun[1][0](np.array([1]))))
# if len(actfun) and len(actfun[0]) and len([1]) == 2:
#     print('seems NOT to work')
#
# g = []
# for i in range(1, 5):
#     g.insert(0, i)
# print(g)
#
# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# x = np.array([9, 5, 3])
# b1 = np.dot(A, x)
# b2 = np.dot(x, A)
# b3 = np.dot(A, np.reshape(x, [3, 1]))
# print(b1)
# print(b2)
# print(b3)
# x2 = np.reshape(x, [3, 1])
# start = time.time()
# for i in range(1000000):
#     np.dot(A, x2)
# end = time.time()
# print('Andra tog ' + str(end - start) + ' s att beräkna')
# start = time.time()
# for i in range(1000000):
#     np.dot(A, x)
# end = time.time()
# print('Första tog ' + str(end - start) + ' s att beräkna')
#
# for j in range(100):
#     A1 = np.random.random_sample([40, 50])
#     x1 = np.random.random_sample([50, 1000])
#     A2 = A1.T
#     x2 = x1.T
#     l1 = []
#     l2 = []
#
#     start1 = time.time()
#     for i in range(10000):
#         np.dot(A1, x1)
#     end1 = time.time()
#     l1.append(end1 - start1)
#
#     start2 = time.time()
#     for i in range(10000):
#         np.dot(x2, A2)
#     end2 = time.time()
#     l2.append(end2 - start2)
#     print(sum(l1) / len(l1))
#     print(sum(l2) / len(l2))
#     print(j)
#     sys.stdout.flush()
# a = np.random.random_sample(np.array([3, 2])) - 1 / 2
# print(a)
# print(np.absolute(a))
# print(nf.L1(a))

# a = [np.random.random_sample(np.array([2, 2]))]
# a.append(np.random.random_sample(np.array([2, 2])))
# print(a)
# j = 0
# for i in [a]:
#     j += 1
#     print(j)
#     print(i)
#
# time1 = 0
# time2 = 0
# for i in range(1000):
#     A = 200 * (np.random.random_sample(np.array([100, 100])) - 1 / 2)
#
#     time2start = time.time()
#     np.power(A, 2)
#     time2end = time.time()
#     time2 += time2end - time2start
#
#     time1start = time.time()
#     np.square(A)
#     time1end = time.time()
#     time1 += time1end - time1start
# print(time1)
# print(time2)

# ANVÄND SQUARE ISTÄLLET FÖR POWER!!! (se ovan)

time1 = 0
time2 = 0
for i in range(5000):
    A = 200 * (np.random.random_sample(np.array([100, 100])) - 1 / 2)

    time1start = time.time()
    nf.sigmoid2(A, True, True)
    time1end = time.time()
    time1 += time1end - time1start

    time2start = time.time()
    nf.sigmoid3(A, True, True)
    time2end = time.time()
    time2 += time2end - time2start

print(time1)
print(time2)
# sigmoid3 snabbare än sigmoid2 ENDAST när både fun och funD = True
