import numpy as np

# \\TODO: ändra ev till tuplar med (funktion, derivata)


def sigmoid(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to sigmoid")
    return np.divide(1, np.add(1, np.exp(-x)))


def sigmoidPrime(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to sigmoidPrime")
    return np.divide(np.exp(x), np.square(np.add(1, np.exp(x))))


def reLU(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to reLU")
    return np.maximum(0, x)


def reLUPrime(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to reLUPrime")
    return np.heaviside(x, 0.5)


def softmax(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to softmax")
    return np.divide(np.exp(x), np.sum(np.exp(x), 0))
# //TODO: Kontrolera att summering är över rätt axel
# //TODO: kan kanske villja göra något med -max eller nått för inte inf


def softmaxPrime(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to softmaxPRIME")
    return np.multiply(np.squre(softmax(x)), np.devide(np.sum(np.exp(x), 0),
                                                       np.exp(x)) - 1)


def linear(x, a=1):
    if type(x) is not np.ndarray or type(a) is not np.ndarray:
        raise TypeError("Wrong input type to linear")
    return np.multiply(np.sum(x, 0), a)  # //TODO: Dimentioner och så...


# def linearPRIME(a=1):
#    if type(a) is not np.ndarray:
#        raise TypeError("Wrong input type to linearPRIME")
#    return a  # //TODO: ÖM... Ja..
