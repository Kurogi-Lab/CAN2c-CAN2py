import chainer
import cupy as cp
#import numpy as np

if __name__ == '__main__':
    x = cp.arange(9, dtype=cp.float32)
    y = cp.zeros_like(x, dtype=cp.float32)
    cp.ElementwiseKernel(
        'raw T x', 'T y',
        '''
        for(int j = 0; j <= i; j++) { 
            y += x[j];
        }
       ''',
        'cumulative_sum')(x, y)
    print(y)
