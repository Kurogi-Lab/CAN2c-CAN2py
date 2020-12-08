import importlib
import numpy as np
#from chainer import cuda
import cupy as cp
#from chainer.cuda import cupy as cp
#cp=importlib.import_module('cupy')
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
#import chainer # This is must, why?
#cp=cuda.cupy
print cp
import time
#n = 3000 #n=3000 np:0.519683122635 cp:0.549144983292
#n = 2000 #n=2000 np:0.505941152573 cp:0.547195911407
#n = 4000 #n=4000 np:0.971707105637 cp:0.50870680809
n = 5000 #n=5000 np:2.14616203308 cp:0.523685932159
#n = 6000 #n=6000 np:2.77858686447 cp:0.538099050522
#n = 7000 #n=7000 np:4.47373604774 cp:0.531967878342
#n = 8000 #n=8000 np:7.09800386429 cp:0.550987005234
#n = 9000 #cupy.cuda.memory.OutOfMemoryError: out of memory to allocate 648000000 bytes (total 1944000000 bytes)

#t1 = time.time()
#a = np.random.rand(n,n)
#b = np.random.rand(n,n)
#np.dot(a,b)
t2 = time.time()
#print ('#n={} np:{}'.format(n,t2-t1))
#t2 = time.time()
a = cp.random.rand(n,n)
b = cp.random.rand(n,n)
cp.dot(a,b)
t3 = time.time()
print ('#n={} cp:{}'.format(n,t3-t2))
#print ('#n={} np:{} cp:{}'.format(n,t2-t1,t3-t2))

