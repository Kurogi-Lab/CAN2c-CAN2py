import numpy as xp
#import cupy as xp
a = xp.random.rand(100)
#B = xp.random.rand(100000, 100)
B = xp.random.rand(1000000, 100) #memory error for cupy
#[numpy.linalg.norm(a - b) for b in B]
xp.linalg.norm(a - B, axis=1)
