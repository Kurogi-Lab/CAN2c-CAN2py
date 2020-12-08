#!/usr/bin/env python
# -*- coding: utf-8 -*-
import importlib

# -------------------------------
""" True: Use CuPy 
    False:Not use CuPy """
#use_cupy=True
#use_cupy = False
try:
  import chainer
  import use_gpu
  use_cupy=True
except:
  use_cupy=False
# -------------------------------

def is_cupy():
    """ Returns: bool: defined this file value. """
    return use_cupy

def xp_factory():
    """ Returns: imported instance of cupy or numpy. """
    if is_cupy():
#        cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
        return importlib.import_module('cupy')
    else:
        return importlib.import_module('numpy')

def xpfloat(xp):
    if is_cupy():
        return xp.float32
    else:
        return xp.float64

def report():
    """ report which is used cupy or numpy. """
    if is_cupy():
        print('import cupy !')
    else:
        print('import numpy !')
