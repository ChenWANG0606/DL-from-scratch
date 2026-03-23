import numpy as np
from scipy.special import erf

def softmax(x):
    x_exp = np.exp(x - np.max(x, axis = -1, keepdims = True))
    return x_exp/np.sum(x_exp,axis = -1,  keepdims = True)

def ReLU(x):
    return np.maximum(0, x)

def ReLU_d(x):
    return np.where(x>0, 1, 0)

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def Sigmoid_d(x):
    s = Sigmoid(x)
    return s*(1-s)


def LeakyReLU(x, alpha):
    return np.maximum(alpha*x, x)

def LeakyReLU_d(x, alpha):
    return np.where(x>0, 1, alpha)

def GeLU(x):
    return 0.5 * x * (1+erf(x/np.sqrt(2)))

def GeLU_d(x):
    cdf = 0.5 *(1 + erf(x/np.sqrt(2)))
    pdf = 1/np.sqrt(2*np.pi) * np.exp(- x**2 / 2)
    return cdf+x*pdf


def Swish(x, beta):
    return x*Sigmoid(beta * x)

def Swish_d(x, beta):
    s = Sigmoid(beta * x)
    return s+beta*x*s*(1-s)


def SwiGLU(x, beta):
    # 拆分输入，假设x最后一位为偶数，拆为两部分 
    x1, x2 = np.split(x, 2, axis = -1)
    swish = Swish(x1, beta)
    return swish * x2


def SwiGLU_d(x, beta):
    x1, x2 = np.split(x, 2, axis = -1)
    deriv =  Swish_d(x, beta) * x2
    return np.concatenate([deriv, x1*s], axis = -1)
    


