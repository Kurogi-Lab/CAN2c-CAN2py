import chainer
import numpy as np
import cupy as cp
import time

#def sigmoid(x):
#    temp = x.copy()
#    for i,x in enumerate(x):
#        temp[i] = 1/(1+math.exp(-x))
#    return temp
#
#start = time.time()
#x = list(range(300000000))
#sigmoid(x)
#print(time.time()-start)

def numpy_sigmoid(x):
    return 1/(1+np.exp(-x))

start = time.time()
x = np.arange(30000000)
numpy_sigmoid(x)
print(time.time()-start)

def cupy_sigmoid(x):
    return 1/(1+cp.exp(-x))

start = time.time()
x = cp.arange(30000000)
numpy_sigmoid(x)
print(time.time()-start)
