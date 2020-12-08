#!/usr/bin/env python
# -*- coding: utf-8 -*-
###
#chainer 3.1.0
##
#see 
# share/am.h AM_VER==1
# share/am.c AM_VER==1
#AM_VER == 1
####
#import cupy as np
#import cupy as sp
##original
#import numpy as np
#import scipy as sp
###both numpy
#import numpy as xp
#import numpy as sp
##both cupy
#import cupy as xp
#import cupy as sp
#import chainer
import switchCupy
xp = switchCupy.xp_factory()
xpfloat=switchCupy.xpfloat(xp)
if switchCupy.is_cupy() == True:
  sp=xp #if switchCupy.is_cupy() == True else importlib.import_module('scipy')
else:
  import scipy as sp
#  sp=importlib.import_module('scipy')
###
import math
#import dask.array as da
#from itertools import product
#import dask

#def init_AM(q,nx,ny):
#  q['nx']=nx
#  q['ny']=ny
#  q['M']=xp.zeros((ny,nx),dtype=xpfloat)
#  q['P']=xp.identity(nx,dtype=xpfloat)*1e4 #modified by kuro 20180111
#  q['x']=xp.zeros(nx,dtype=xpfloat)
#  q['y']=xp.zeros(ny,dtype=xpfloat)
#
##  init_AMdata(q)
#  return #def init_AM(q,nx,ny):

def init_AM(q,nc,nx,ny): 
  q['nx']=nx
  q['ny']=ny
  q['nc']=nc
  q['M']=xp.zeros((nc,ny,nx),dtype=xpfloat)
  q['P']=xp.zeros((nc,nx,nx),dtype=xpfloat)
#  q['P']=xp.identity(nx,dtype=xpfloat)*1e4 #modified by kuro 20180111
  q['x']=xp.zeros((nc,nx),dtype=xpfloat)
  q['y']=xp.zeros((nc,ny),dtype=xpfloat)
#  for n,i in itertools.product(xrange(nc),xrange(nx)):
#    q['P'][n,i,i]=1e4
  for n in xrange(nc):
    for i in xrange(nx):
      q['P'][n,i,i]=1e4 #modified by kuro 20180111
#  q['P']=xp.identity(nx,dtype=xpfloat)*1e4 #modified by kuro 20180111
  q['x']=xp.zeros((nc,nx),dtype=xpfloat)
  q['y']=xp.zeros((nc,ny),dtype=xpfloat)
#  import pdb;pdb.set_trace() #for debug 
#  init_AMdata(q)
  return #def init_AM(q,nx,ny):

#def free_AM(q):
#  del q


#連想行列Mの更新 #AM_VER==1
def calc_AM(q,i):
  nx=q['nx']
  ny=q['ny']

#numpy
  p=xp.zeros(nx,dtype=xpfloat)
  g=xp.zeros(nx,dtype=xpfloat)

#dask
#  q['P_spirit']=da.from_array(xp.array(q['P']),chunks=int(nx/2))
#  q['x_spirit']=da.from_array(xp.array(q['x']),chunks=int(nx/2))

#########################
#original
#  for i in xrange(nx):
#    g[i]=0
#    for j in xrange(nx):
#      g[i]+= q['P'][i,j]*q['x'][j]

#numpy
  g=sp.dot(q['P'][i], q['x'][i])

#daskを使ってみた
#  g_res=da.dot(q['P_spirit'], q['x_spirit'])		#分割して計算（この時点では計算されてない）
#  g=g_res.compute()					#結合（ここで計算）
#  import pdb;pdb.set_trace(); #for debug 

###########################
#original
#  xPx1=1.0
#  for i in xrange(nx):
#    xPx1 += q['x'][i]*g[i]
#  for i in xrange(nx):
#    p[i]=g[i]/xPx1

#numpy
  xPx1=float(sp.dot(q['x'][i],g))
  xPx1+=1.0
  p=g/xPx1


###########################

#original
#  for i in xrange(nx):
#    for j in xrange(nx):
#      q['P'][i,j] -= p[i]*g[j]

#  for i in xrange(ny):
#    err=q['y'][i]
#    for j in xrange(nx):
#      err -= (q['M'][i,j]*q['x'][j])
#    for j in xrange(nx):
#      q['M'][i,j] += err*p[j]

#numpy 
#  import pdb;pdb.set_trace() #for debug 
  for j in xrange(nx):
    q['P'][i,j] = q['P'][i,j] - p[j]*g  #p[j]:scalar
#    q['P'][i,j] = q['P'][i,j] - sp.dot(p[j],g)
#  q['P'][i] = q['P'][i] - sp.dot(p,g)
  for j in xrange(ny):
#    import pdb;pdb.set_trace() #for debug 
    err = q['y'][i,j] - sp.dot(q['M'][i,j],q['x'][i])
    q['M'][i,j] = q['M'][i,j] + (err*p)

#  q['P'] = [(q['P'][i] - sp.dot(p[i],g)).tolist for i in xrange(nx)]				#論文式(15)更新式
#  q['M'] = [q['M'][i] + ((q['y'][i] - sp.dot(q['M'][i],q['x'])) * p) for i in xrange(ny)]	#論文式(14)更新式
#  import pdb;pdb.set_trace(); #for debug 
#  for i in xrange(ny):
#    err = q['y'][i] - sp.dot(q['M'][i],q['x'])
#    q['M'][i] += (q['y'][i] - sp.dot(q['M'][i],q['x'])) * p


#  for i in xrange(ny):
#    err = q['y'][i] - sp.dot(q['M'][i],q['x'])
#    q['M'][i] += err * p


############################
#  del p
#  del g   
  return #def calc_AM(q):

def calc_AMxy(q,i,xi,yi):
  nx=q['nx']
  ny=q['ny']

#numpy
  p=xp.zeros(nx,dtype=xpfloat)
  g=xp.zeros(nx,dtype=xpfloat)

#numpy
  g=sp.dot(q['P'][i], xi)

#numpy
  p=g/(float(sp.dot(xi,g))+1.0)

#numpy 
#  import pdb;pdb.set_trace() #for debug 
  q['P'][i] = q['P'][i] - (p.reshape(nx,1))*(g.reshape(1,nx))  #p[j]:scalar
#  for j in xrange(nx):
#    q['P'][i,j] = q['P'][i,j] - p[j]*g  #p[j]:scalar,g vector

#  import pdb;pdb.set_trace() #for debug 
#  q['M'][i] = q['M'][i] + ((yi - sp.dot(q['M'][i],xi))*p)         #works for any ny, but slower?
  q['M'][i,0] = q['M'][i,0] + ((yi[0] - sp.dot(q['M'][i,0],xi))*p) #works for ny=1
#  for j in xrange(ny):                                            #works and faster?
#    q['M'][i,j] = q['M'][i,j] + ((yi[j] - sp.dot(q['M'][i,j],xi))*p)

  return #def calc_AM(q):

