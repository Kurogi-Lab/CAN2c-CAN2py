#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cupy as cp
import numpy as np

if __name__ == '__main__':
    x = cp.arange(10, dtype=np.float32).reshape(2,5)
    #
    L2norm_kernel = cp.ReductionKernel(
        'T x',      # in_params:入力
        'T y',      # out_params:出力
        ' x * x',   #map_expr:前処理
        'a + b',    #reduce_expr:リデュース
        'y = sqrt(a)',  #post_map_expr:後処理
        '0',        #identity:初期値
        'l2norm')   #name:名前
    y = L2norm_kernel(x)
    print(x)
    print(y)

##from p.13 of https://www.slideshare.net/ryokuta/cupy
#import chainer
#import cupy as cp
##import numpy as np
#
#l2norm_kernel = cp.ReductionKernel(
#  'T x', #input
#  'T y', #output
#  'x * x', #preprocess
#  'a + b',  #reduce ?
#  'y=sqrt(a)', #post-process
#  '0',         #initial-value
#  'l2norm')    #name
#x=cp.arange(10,dtype='f').reshape(2,5)
#y=l2norm_kernel(x,axis=1, keepdims=True)
#print('x={}, y={}'.format(x,y))
