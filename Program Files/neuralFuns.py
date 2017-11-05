import numpy as np

# \\TODO: ändra ev till tuplar med (funktion, derivata)

# Actfuns


def sigmoid(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to sigmoid")
    return np.divide(1, np.add(1, np.exp(-x)))


def sigmoidPrime(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to sigmoidPrime")
    return np.divide(np.exp(x), np.square(np.add(1, np.exp(x))))


def sigmoid2(x, fun=True, funD=False):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to sigmoid2")
    ret = ()
    if fun:
        ret += (np.divide(1, np.add(1, np.exp(-x))), )
    if funD:
        ret += (np.divide(np.exp(x), np.square(np.add(1, np.exp(x)))), )
    return ret


def sigmoid3(x, fun=True, funD=False):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to sigmoid2")
    ret = ()
    funeval = np.divide(1, np.add(1, np.exp(-x)))
    if fun:
        ret += (funeval, )
    if funD:
        ret += (np.multiply(np.exp(x), np.square(funeval)), )
    return ret


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

# Compfuns
# TODO, should perhaps also work for list of ndarrays

def L1(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to L1")
    return np.sum(np.absolute(x))


def L1Prime(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to L1Prime")
    return np.sign(x)


def L2(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to L1")
    return np.multiply(np.divide(1, 2), np.sum(np.square(x)))


def L2Prime(x):
    if type(x) is not np.ndarray:
        raise TypeError("Wrong input type to L1")
    return x
